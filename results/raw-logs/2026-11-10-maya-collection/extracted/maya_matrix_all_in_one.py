#!/usr/bin/env python3
"""Single-file Maya HIP benchmark. Writes only to a timestamped /opt/maya-test-results run.

Uses two existing Maya configs; derives configurations at runtime. Built-in --bench for
2K/8K; original API payloads for 32K/64K. No model downloads or source edits.
"""
import argparse, csv, json, os, shutil, signal, socket, subprocess, sys, time
import urllib.request, urllib.error
from datetime import datetime
from pathlib import Path

MAYA = Path('/opt/project-maya')
BASE = Path('/opt/maya-test-results')
REQUESTS = BASE / '20261028-223138/requests'
SOURCES = {'deepseek': MAYA/'maya-ds4-production.json',
           'glm': MAYA/'maya-ud-q4_k_xl-hip.json'}
STOP = False

def interrupted(*_):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)

def flag(args, name, value):
    if name in args:
        i = args.index(name)
        if i+1 >= len(args): raise ValueError('Missing value after '+name)
        args[i+1] = str(value)
    else: args.extend((name, str(value)))

def remove_flag(args, name, takes_value=False):
    while name in args:
        i=args.index(name); del args[i:i+(2 if takes_value else 1)]

def port_open(port):
    with socket.socket() as s:
        s.settimeout(.4)
        return s.connect_ex(('127.0.0.1', port)) == 0

def process(cmd, log, env=None):
    handle=open(log,'wb',buffering=0)
    p=subprocess.Popen(cmd,cwd=MAYA,stdout=handle,stderr=subprocess.STDOUT,
                       env=env,start_new_session=True)
    return p,handle

def stop(p):
    if p.poll() is not None:return
    try: os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)

def prepare(model, gpu_count, mode, source, port, reserve, extra_reserve):
    c=json.loads(source.read_text())
    if c.get('backend')!='hip':raise ValueError('Expect HIP config: '+str(source))
    c['gpu']=list(range(gpu_count))
    c['port']=port
    c['args']=list(c['args'])
    a=c['args']
    # Reserve is NOT a hard cap on total VRAM or exact routed expert pool size.
    # 'reduced' asks Maya to leave more VRAM free than 'full'; quantify actual
    # allocations via GPU samples and engine startup log.
    pad = extra_reserve if mode=='reduced' else 0
    if model=='deepseek':
        flag(a,'--tp',2 if gpu_count==4 else 1)
        flag(a,'--reserve-mib', reserve+pad)
        if mode=='zero':
            if '--no-expert-cache' not in a:a.append('--no-expert-cache')
        else:remove_flag(a,'--no-expert-cache')
    else:
        flag(a,'--vram-reserve-mib',reserve+pad)
        flag(a,'--vram-reserve-later-mib',reserve+pad)
        # env MAYA_GLM_RESERVE_MB can conflict with explicit args: remove it.
        c.setdefault('env',{}).pop('MAYA_GLM_RESERVE_MB',None)
        c['env']['MAYA_GLM_SPLIT']='auto' if gpu_count==4 else '0'
        if mode=='zero':
            if gpu_count!=1: raise ValueError('GLM exact zero cache on 4 GPUs unsupported: '
                                                '--expert-cache 0 only controls first GPU')
            flag(a,'--expert-cache','0')
        else:flag(a,'--expert-cache','auto')
    # Don't put new config/logs in project root
    return c

def api(c,payload=None,timeout=10):
    uri=f"http://127.0.0.1:{c['port']}/v1/" + ('models' if payload is None else 'chat/completions')
    headers={'Content-Type':'application/json'}
    if c.get('api_key'):headers['Authorization']='Bearer '+c['api_key']
    req=urllib.request.Request(uri,data=payload,headers=headers)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req,timeout=timeout) as r:return r.read()

def wait_ready(c,p,seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if STOP:raise KeyboardInterrupt()
        if p.poll() is not None:raise RuntimeError(f'engine exited {p.returncode}')
        try: api(c,timeout=3);return
        except Exception: time.sleep(2)
    raise TimeoutError('engine did not start; inspect server.log')

def run_command(cmd,log):
    p,h=process(cmd,log)
    try:
        while p.poll() is None:
            if STOP:raise KeyboardInterrupt()
            time.sleep(.5)
        if p.returncode:raise RuntimeError(f'Exit {p.returncode}; inspect {log}')
    finally:stop(p);h.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',nargs='+',choices=['deepseek','glm'],default=['deepseek','glm'])
    ap.add_argument('--gpus',nargs='+',type=int,choices=[1,4],default=[4,1])
    ap.add_argument('--modes',nargs='+',choices=['full','reduced','zero'],default=['full','reduced','zero'])
    ap.add_argument('--reserve-mib',type=int,default=1024,
                    help='baseline per GPU; 1024 MiB default')
    ap.add_argument('--extra-reserve-mib',type=int,default=8192,
                    help='ADDITIONAL reserve in reduced mode; approximate capacity reduction, not exact half')
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--port',type=int,default=18081)
    ap.add_argument('--load-timeout',type=int,default=1200)
    ap.add_argument('--request-timeout',type=int,default=2400)
    ap.add_argument('--skip-short',action='store_true')
    ap.add_argument('--skip-long',action='store_true')
    ap.add_argument('--no-gpu-samples',action='store_true')
    ap.add_argument('--yes',action='store_true',help='run without interactive confirmation')
    ap.add_argument('--continue-on-error',action='store_true')
    args=ap.parse_args()
    if args.repeats<1 or args.reserve_mib<0 or args.extra_reserve_mib<0:ap.error('invalid counts/reserves')
    if port_open(args.port):ap.error(f'port {args.port} already in use')
    for model in args.models:
        if not SOURCES[model].is_file():ap.error('Missing '+str(SOURCES[model]))
    if not args.skip_long:
        for n in (32768,65536):
            p=REQUESTS/f'r1-{n}.json'
            if not p.is_file():ap.error('Missing original long request '+str(p))
    jobs=[]; skips=[]
    for model in args.models:
        for gpu_count in args.gpus:
            for mode in args.modes:
                name=f'{model}-{gpu_count}gpu-{mode}'
                if model=='glm' and gpu_count==4 and mode=='zero':
                    skips.append((name,'GLM --expert-cache 0 only controls GPU 0'))
                else:jobs.append((name,model,gpu_count,mode))
    print('\nSource configs (read-only):', *map(str,(SOURCES[m] for m in args.models)),sep='\n  ')
    print('Run order:',', '.join(j[0] for j in jobs))
    print('Skipped:',skips)
    print('REDUCED means increased reserve, NOT mathematically half the expert cache.')
    print('Short benchmarks use 2K/8K; long use original 32K/64K requests.')
    print('Stop all existing Maya servers first; no model downloads occur.')
    if not args.yes and input('Proceed? [y/N] ').lower() not in ('y','yes'):return 0
    run=BASE/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-maya-matrix')
    run.mkdir(parents=True,exist_ok=False,mode=0o700)
    run.chmod(0o700)
    (run/'notes.json').write_text(json.dumps({'options':vars(args),'skipped':skips},indent=2)+'\n')
    print('Results:',run,flush=True)
    fields=['variant','model','gpus','mode','context','repeat','prompt_n','cache_n',
            'prefill_tps','predicted_n','decode_tps','status','error']
    with (run/'results.csv').open('w',newline='') as f:csv.DictWriter(f,fields).writeheader()
    for name,model,gpu_count,mode in jobs:
        if STOP:break
        d=run/name;d.mkdir()
        print('\nRUN',name,flush=True)
        try:
            c=prepare(model,gpu_count,mode,SOURCES[model],args.port,
                      args.reserve_mib,args.extra_reserve_mib)
            c['log']=str(d/'maya-engine.log')
            cp=d/'config.json';cp.write_text(json.dumps(c,indent=2)+'\n'); cp.chmod(0o600)
            launcher=[str(MAYA/'maya.sh'),'--backend','hip','--config',str(cp),'--plain']
            if not args.skip_short:
                run_command(launcher+['--bench'],d/'short.log')
                for fn in ('maya-bench.txt','maya-bench.log'):
                    source=MAYA/fn
                    if source.exists():shutil.move(str(source),str(d/fn))
            if args.skip_long:continue
            if port_open(args.port):raise RuntimeError('Port occupied before start')
            server,h=process(launcher,d/'server.log')
            sample,sh=None,None
            try:
                wait_ready(c,server,args.load_timeout)
                if not args.no_gpu_samples and shutil.which('rocm-smi'):
                    sample,sh=process(['bash','-c',
                        'while true; do date -Is; rocm-smi --showuse --showpower --showmemuse; sleep 2; done'],
                        d/'gpu.log')
                for n in (32768,65536):
                    original=json.loads((REQUESTS/f'r1-{n}.json').read_text())
                    original['model']=c['model_name']
                    raw=json.dumps(original).encode('utf-8')
                    for repetition in range(1,args.repeats+1):
                        if STOP:raise KeyboardInterrupt()
                        row={'variant':name,'model':model,'gpus':gpu_count,
                             'mode':mode,'context':n,'repeat':repetition}
                        try:
                            response=api(c,raw,args.request_timeout)
                            (d/f'response-{n}-r{repetition}.json').write_bytes(response)
                            result=json.loads(response)
                            if result.get('error'):raise RuntimeError(str(result['error']))
                            t=result.get('timings') or {}
                            row.update(prompt_n=t.get('prompt_n',''),cache_n=t.get('cache_n',''),
                                       prefill_tps=t.get('prompt_per_second',''),
                                       predicted_n=t.get('predicted_n',''),
                                       decode_tps=t.get('predicted_per_second',''),status='ok')
                            if row['cache_n'] not in (0,'0'):
                                row['status']='cached-not-comparable'
                            print(' ',n,'repeat',repetition,'prefill',row['prefill_tps'],
                                  'decode',row['decode_tps'],'cache',row['cache_n'],flush=True)
                        except Exception as exc:
                            row.update(status='error',error=str(exc))
                            raise
                        finally:
                            with (run/'results.csv').open('a',newline='') as f:
                                csv.DictWriter(f,fields).writerow(row)
            finally:
                if sample is not None:stop(sample);sh.close()
                stop(server);h.close()
        except (Exception,KeyboardInterrupt) as exc:
            (d/'ERROR.txt').write_text(repr(exc)+'\n')
            print('FAILED',name,repr(exc),file=sys.stderr,flush=True)
            if not args.continue_on_error:return 1
    return 130 if STOP else 0

if __name__=='__main__':sys.exit(main())
