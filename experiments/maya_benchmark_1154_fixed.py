#!/usr/bin/env python3
"""Unattended, isolated Project Maya v1.1.54 DeepSeek HIP experiment runner.
Run with Maya stopped: python3 maya_benchmark_1154.py /path/to/project-maya /path/to/maya-ds4-hip.json
Never writes to the production config; never downloads models or changes sampling there.
"""
import copy, csv, datetime, json, os, pathlib, re, shutil, subprocess, sys, time, traceback, statistics, signal

TAG = 'v1.1.54'
SIZES = (8192, 32768, 65536)

def cmd(argv, cwd, log=None):
    with (open(log,'a') if log else open(os.devnull,'w')) as f:
        p=subprocess.run(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=False)
    if p.returncode: raise RuntimeError(f'Command failed ({p.returncode}): {argv}; log: {log}')

def main():
    if len(sys.argv)!=3:
        sys.exit('Usage: python3 maya_benchmark_1154.py MAYA_REPOSITORY ORIGINAL_CONFIG_JSON')
    root=pathlib.Path(sys.argv[1]).resolve()
    source=pathlib.Path(sys.argv[2]).resolve()
    if not (root/'maya.py').exists() or not source.exists(): sys.exit('Repository or config not found')
    if not (root/'.git').exists(): sys.exit('Expected a Git checkout of Project Maya')
    initial=json.loads(source.read_text())
    if initial.get('backend')!='hip' or '--pack' not in initial.get('args',[]):
        sys.exit('This script requires an existing DeepSeek HIP config with --pack')
    if '--max-context' not in initial['args'] or int(initial['args'][initial['args'].index('--max-context')+1]) < 65536:
        sys.exit('Original --max-context must be >=65536 (no automatic production config edits)')
    # Allow the user to copy this standalone, untracked script into the repo.
    # Never overwrite staged or unstaged modifications to tracked source.
    for diff_args in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
        if subprocess.run(diff_args,cwd=root).returncode:
            sys.exit('Tracked Git changes found. Commit/stash changes before unattended upgrade.')
    # Refuse to compete with any existing Maya process (do not kill it).
    ps=subprocess.run(['ps','-eo','args='],capture_output=True,text=True,check=True).stdout
    if any(('maya-ds4' in x and '--serve' in x) for x in ps.splitlines() if 'ps -eo' not in x):
        sys.exit('A maya-ds4 server appears to be running. Stop it before benchmarking.')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out=root.parent/f'maya-bench-1154-{stamp}'
    out.mkdir(mode=0o700)
    (out/'original-config.json').write_text(json.dumps(initial,indent=2))
    (out/'previous-revision.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True))
    log=out/'update.log'
    print('Results:',out,flush=True)
    cmd(['git','fetch','--tags','origin',TAG],root,log)
    sha=subprocess.check_output(['git','rev-parse','--verify',f'refs/tags/{TAG}^{{commit}}'],cwd=root,text=True).strip()
    cmd(['git','switch','--detach',sha],root,log)
    # Use the same build refresh routine as Maya's normal start. No model download or setup.
    sys.path.insert(0,str(root))
    os.chdir(root)
    import maya as M
    import serve.server as SV
    import strata_tokenizer as ST
    M.select_build_backend('hip')
    with open(log,'a') as f:
        oldout,olderr=sys.stdout,sys.stderr
        try:
            sys.stdout=sys.stderr=f
            M.refresh_engine(initial)
        finally: sys.stdout,sys.stderr=oldout,olderr
    if not M.bench_exe(initial).exists(): raise RuntimeError('HIP engine binary missing after update')
    tokdir=pathlib.Path(initial['tokenizer'])
    tok=ST.Tokenizer.from_dir(tokdir)
    # Repeat a fixed body of text to achieve long token counts; different prefixes prevent KV reuse.
    docs='\n\n'.join(p.read_text(errors='replace') for p in [root/'README.md',*sorted((root/'docs').glob('*.md')),*sorted((root/'bench'/'results').glob('*.md'))] if p.exists())
    ids=tok.encode(docs)
    if not ids: raise RuntimeError('No source tokens available to form benchmark prompts')
    rows=[]
    # Always remove inherited overrides so 'default' is a real default, even when the
    # parent shell has been used for manual Maya tuning previously.
    tested_vars = (
        'DS4_TP_SHARES', 'DS4_AR_P2P', 'DS4_AR_P2P_LOG',
        'MAYA_DS4_HIP_PEER_AR', 'DS4_FLOW_COPY_DMA', 'DS4_AI_RUN',
        'DS4_CHUNK_AUTO', 'DS4_YOCO_FLOW', 'DS4_LENT_RAM_GB',
        'MAYA_DS4_EXPERT_MMQ', 'MAYA_DS4_PREFILL_FUSED',
    )
    args=initial['args']
    def argval(opt,default=None):
        return args[args.index(opt)+1] if opt in args and args.index(opt)+1<len(args) else default
    gpu_value=argval('--gpus','') or argval('--gpu','')
    if isinstance(initial.get('gpu'),(list,tuple)):
        gpu_count=len(initial['gpu'])
    elif initial.get('gpu') is not None:
        gpu_count=len(str(initial['gpu']).split(','))
    elif '--gpus' in args:
        at=args.index('--gpus')+1
        gpu_tokens=[]
        while at<len(args) and not str(args[at]).startswith('--'):
            gpu_tokens.append(str(args[at])); at+=1
        gpu_count=len(re.findall(r'\d+', ' '.join(gpu_tokens)))
    else:
        gpu_count=len(re.findall(r'\d+',str(gpu_value))) if gpu_value else 0
    # In Maya, --tp 2 on four GPUs means two successive pipeline groups (a layer split).
    tp=int(argval('--tp','1'))
    group_count=(gpu_count+tp-1)//tp if gpu_count else 0
    # If --gpus is represented unusually, do not silently claim flow eligibility.
    flow_eligible=group_count>1
    # These options are actually documented in the v1.1.54 SETTINGS.md.
    environments=[
        ('default',{},'baseline'),
        ('tp_even',{'DS4_TP_SHARES':'off'},'placement'),
        ('p2p_generic',{'DS4_AR_P2P':'1','DS4_AR_P2P_LOG':'1'},'transport'),
        ('p2p_hip_auto',{'MAYA_DS4_HIP_PEER_AR':'auto'},'transport'),
    ]
    if flow_eligible:
        environments += [
            ('flow_dma',{'DS4_FLOW_COPY_DMA':'1'},'flow'),
            ('flow_ai',{'DS4_AI_RUN':'1'},'flow'),
            ('flow_chunk',{'DS4_CHUNK_AUTO':'1'},'flow_approx'),
            ('flow_yoco',{'DS4_YOCO_FLOW':'1'},'flow_approx'),
            ('flow_combined',{'DS4_FLOW_COPY_DMA':'1','DS4_AI_RUN':'1','DS4_CHUNK_AUTO':'1','DS4_YOCO_FLOW':'1'},'flow_approx'),
        ]
    # Q8_K_XL commonly uses native MXFP4 experts. The opt-in only covers IQ experts;
    # run it as a negative-control experiment and determine whether it activated.
    environments.append(('expert_mmq',{'MAYA_DS4_EXPERT_MMQ':'1'},'quantized_approx'))
    # A separate memory-tier experiment only if the production setup enables borrowing.
    inherited={**os.environ,**(initial.get('env') or {})}
    if str(inherited.get('DS4_BORROW','')).lower() not in ('0','off','false') and 'DS4_BORROW' in inherited:
        environments.append(('lent_ram_8g',{'DS4_LENT_RAM_GB':'8'},'borrow'))
    skipped=[]
    if not flow_eligible: skipped.append(f'Flow variants skipped: could not establish >1 GPU group from config GPUs={gpu_count}, --tp={tp}.')
    if not any(n=='lent_ram_8g' for n,_,_ in environments):
        skipped.append('Lent-RAM variant skipped: DS4_BORROW not explicitly enabled in config/environment.')
    skipped.append('MAYA_PROMPT_MEMO is a server-level tokenization optimization and cannot be measured by the engine benchmark protocol.')
    skipped.append('MMQ is attempted on Q8_K_XL but may be a no-op: documented only for supported IQ expert shapes. No quality parity is assumed.')
    (out/'skipped.txt').write_text('\n'.join(skipped)+'\n')
    (out/'experiment-plan.json').write_text(json.dumps([dict(name=n,overrides=e,category=c) for n,e,c in environments],indent=2))
    print('Starting',len(environments),'variants, 8K/32K/64K, two repeats each',flush=True)

    for name,extra,category in environments:
        variant=copy.deepcopy(initial)
        variant['env']=dict(variant.get('env') or {})
        # Explicitly clear all tested variables to 0 in config so a later variant
        # cannot inherit a previous override from the caller environment.
        for key in tested_vars:
            variant['env'].pop(key,None)
        variant['env'].update(extra)
        # This is an *engine-only* launch; ensure inherited parent overrides
        # cannot contaminate child_env() when Maya constructs the child process.

        (out/f'{name}.config.json').write_text(json.dumps(variant,indent=2))
        logpath=out/f'{name}.engine.log'
        print(f'[{name}] {extra}',flush=True)
        try:
          with open(logpath,'w') as lf:
              command=[str(M.bench_exe(variant)),'--serve',*SV.engine_args(variant)]
              child=SV.child_env(variant)
              for key in tested_vars:
                  child.pop(key,None)
              child.update(extra)
              proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=lf,text=True,cwd=variant.get('cwd') or str(root),
                                    env=child,bufsize=1)

              try:
                  ready=False; multi=False
                  while True:
                      line=proc.stdout.readline()
                      if not line: raise RuntimeError(f'{name}: engine exited before READY')
                      if line.startswith('READY'):
                          ready=True; multi=any(x.startswith('multi=') for x in line.split()); break
                  def run(tokens,n_new=1):
                      proc.stdin.write(M.bench_gen_line(n_new,tokens,multi));proc.stdin.flush()
                      while True:
                          line=proc.stdout.readline()
                          if not line: raise RuntimeError(f'{name}: engine exited during test')
                          f=M.bench_reply(line,multi)
                          if f is not None:
                              if f[0]=='ERR': raise RuntimeError(f'{name}: engine ERR: {line.strip()}')
                              return f
                  # A decode benchmark, separately reported. Deterministic benchmark setting is not saved.
                  run(tok.encode(M.bench_chat('deepseek',tokdir)('Say hello in five languages.'),parse_special=True),64)
                  for i in range(2):
                      f=run(tok.encode(M.bench_chat('deepseek',tokdir)(M.BENCH_TOPICS[i]),parse_special=True),256)
                      if float(f[4])>0:
                          rows.append(dict(variant=name,kind='decode',tokens=int(f[1]),repeat=i+1,
                                           tokens_per_s=round(int(f[1])*1000/float(f[4]),2),status='ok'))
                  # Warm a prompt once; use changing leading tokens for each measurement to avoid reuse.
                  run(ids[-8192:],1)
                  for n in SIZES:
                      for r in range(2):
                          offset=((n//1024)*997+r*131)%len(ids)
                          prompt=(ids[offset:]+ids[:offset])
                          tokens=(prompt*((n//len(prompt))+1))[:n]
                          f=run(tokens,1)
                          if len(f)<4 or float(f[3])<=0: raise RuntimeError(f'Bad timing: {f}')
                          rate=n*1000/float(f[3])
                          record=dict(variant=name,kind='prefill',tokens=n,repeat=r+1,
                                      tokens_per_s=round(rate,2),status='ok')
                          rows.append(record)
                          print(record,flush=True)
              finally:
                  if proc.poll() is None:
                      try: proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=60)
                      except Exception: proc.kill();proc.wait()
        except Exception as exc:
            rows.append(dict(variant=name,kind='error',tokens=0,repeat=0,
                             tokens_per_s='',status=f'FAILED: {str(exc)[:240]}'))
            print(f'[{name}] FAILED: {exc} (continuing to next variant)',flush=True)
        with open(out/'results.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['variant','kind','tokens','repeat','tokens_per_s','status'])
            writer.writeheader();writer.writerows(rows)
    summary=['Project Maya v1.1.54 DeepSeek HIP benchmarks','Results:']
    for name,_,_ in environments:
        for n in SIZES:
            rates=[x['tokens_per_s'] for x in rows if x['variant']==name and x['kind']=='prefill' and x['tokens']==n]
            if rates: summary.append(f'{name:10s} {n:6d} tokens: {statistics.mean(rates):8.1f} t/s (n={len(rates)})')
    summary.append('Transport verification (log-based, not hardware-counter proof):')
    for name,_,category in environments:
        if category != 'transport': continue
        logfile=out/f'{name}.engine.log'
        if logfile.exists():
            matches=[line.strip()[:300] for line in logfile.read_text(errors='replace').splitlines()
                     if re.search(r'peer|p2p|all.reduce|host.mapped|BAR1|rebar',line,re.I)]
            summary.append(f'  {name}: '+(' | '.join(matches[:8]) if matches else 'NO P2P/PEER CONFIRMATION FOUND; do not claim enabled'))
    summary.append('Approximate variants (flow_chunk, flow_yoco, flow_combined, expert_mmq) require numerical quality validation before deployment.')
    summary.append('The MMQ flag on Q8_K_XL may be a no-op; compare logs/kernels before attributing a speed difference to MMQ.')
    summary.append('No original configuration was modified. Production engine remains stopped after benchmarking.')
    (out/'summary.txt').write_text('\n'.join(summary)+'\n')
    print('\n'.join(summary),flush=True)

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
