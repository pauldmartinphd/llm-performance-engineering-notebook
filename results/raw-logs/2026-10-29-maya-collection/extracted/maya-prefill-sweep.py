 #!/usr/bin/env python3
"""Maya prefill sweep. Run with the Maya checkout's .venv/bin/python.
Uses a read-only existing build/pack and isolated configs, logs and HTTP port.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flag(args, name, value):
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])


def value(args, name):
    return args[args.index(name) + 1] if name in args else None


def port_busy(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def engines():
    found = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if (p / "comm").read_text().strip() in ("maya-ds4", "llama-server", "maya", "maya-mm3"):
                found.append(p.name)
        except (OSError, ProcessLookupError):
            pass
    return found


def capture(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {"command": cmd, "exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except Exception as e:
        return {"command": cmd, "error": str(e)}


def stop_group(p):
    # The engine inherits the dedicated process group, even if the HTTP parent exits.
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maya", default="/opt/project-maya")
    ap.add_argument("--config", default="maya-ds4-hip.json")
    ap.add_argument("--out", default="/opt/maya-test-results")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--chunks", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--rings", type=int, nargs="+", default=[128, 400, 800])
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--startup-timeout", type=int, default=1200)
    ap.add_argument("--request-timeout", type=int, default=1200)
    ap.add_argument("--tp", type=int, help="Optional separate sweep with a fixed TP value")
    ap.add_argument("--cpu-threads", type=int, help="Optional separate sweep with a fixed CPU-thread count")
    ap.add_argument("--corpus", type=Path, help="Optional UTF-8 document; otherwise use this checkout's docs")
    a = ap.parse_args()
    if sys.platform != "linux":
        ap.error("Run this on the Linux Maya host, not your Mac.")
    if a.port == 8081:
        ap.error("8081 is reserved for the daily-use server.")
    if any(n <= 0 for n in a.chunks + a.rings + a.lengths) or a.repeats < 1:
        ap.error("Sizes and repeats must be positive.")
    if any(n > 32768 or n % 16 for n in a.chunks):
        ap.error("v1.1.30 chunk sizes must be multiples of 16, at most 32768.")
    if port_busy(8081) or port_busy(a.port) or engines():
        ap.error("Stop Maya on 8081 and any other inference engines first. No processes were stopped.")
    root = Path(a.maya).resolve()
    cp = Path(a.config)
    cp = cp if cp.is_absolute() else root / cp
    cfg = json.loads(cp.read_text())
    exe = Path(cfg["exe"])
    if not exe.is_file() or not (root / "serve/server.py").is_file():
        ap.error("Existing engine or server source is missing; build Maya first.")
    ctx = int(value(cfg["args"], "--max-context") or 32768)
    if max(a.lengths) + 512 > ctx:
        ap.error("The config context must exceed the largest test prompt by at least 512 tokens.")
    out = Path(a.out).resolve() / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    os.chmod(out, 0o700)
    sys.path.insert(0, str(root / "tools"))
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_dir(Path(cfg["tokenizer"]))
    corpus_files = [a.corpus.resolve()] if a.corpus else [root / "README.md"] + sorted((root / "docs").glob("*.md"))
    corpus = "\n\n".join(p.read_text(errors="replace") for p in corpus_files)
    if not corpus.strip():
        raise RuntimeError("Empty corpus")
    (out / "corpus.txt").write_text(corpus)
    tokens = tok.encode(corpus)
    while len(tokens) < max(a.lengths) + 4096:
        tokens += tokens
    prompts = []
    for rep in range(a.repeats):
        for n in a.lengths:
            # Different opening for each request prevents a shared long prefix.
            # The exact same request files are reused in every configuration.
            prefix = f"Document trial {rep + 1}, target length {n}. Read the following document.\n"
            offset = (rep * 997 + n // 8) % 3000
            body = tok.decode(tokens[offset:offset + n])
            text = prefix + body + "\nSummarize the document's technical points concisely."
            prompts.append((rep + 1, n, text))
    rng = random.Random(20261028)
    rng.shuffle(prompts)
    pd = out / "requests"
    pd.mkdir()
    for rep, n, text in prompts:
        write_json(pd / f"r{rep}-{n}.json", {"model": cfg.get("model_name", "deepseek-v4.1-flash"),
            "messages": [{"role": "user", "content": text}], "stream": False,
            "temperature": 0, "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False}})
    pairs = list(itertools.product(a.chunks, a.rings))
    baseline = (8192, 400) if (8192, 400) in pairs else pairs[0]
    pairs.remove(baseline)
    rng.shuffle(pairs)
    pairs = [baseline] + pairs + [baseline]  # final baseline checks time/order drift
    clean = copy.deepcopy(cfg)
    for key in ("api_key", "mcp_servers", "cors_origins"):
        clean.pop(key, None)
    write_json(out / "source-config-redacted.json", clean)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "options": vars(a) | {"corpus": str(a.corpus)},
        "source_config_sha256": sha(cp), "script_sha256": sha(Path(__file__)),
        "engine_sha256": sha(exe), "corpus_sha256": sha(out / "corpus.txt"),
        "cases": pairs, "note": "Token lengths are document targets; actual API prompt counts are in results.csv. "
        "Two repeats share one engine start. DSpark and vision are retained from the source config. "
        "First timed request has 128 output tokens, as do all other timed requests. No TTFT is inferred from wall time."}
    meta["commands"] = [capture(cmd) for cmd in [
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
        ["git", "-C", str(root / "third_party/llama.cpp"), "rev-parse", "HEAD"],
        ["uname", "-a"], ["lscpu"], ["free", "-h"],
        ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--showuse"],
        [str(root / ".venv/bin/python"), "-m", "pip", "freeze"]]]
    import resource
    meta["memlock_limits"] = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    pack = Path(value(cfg["args"], "--pack"))
    meta["model_inventory"] = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                               for p in sorted(pack.parent.glob("*.gguf"))]
    meta["model_inventory_note"] = "Size/time inventory only; model weights were not hashed or modified."
    write_json(out / "metadata.json", meta)
    print(f"Results: {out}\n{len(pairs)} starts; {len(prompts)} timed requests per start. Allow several hours.", flush=True)
    fields = ["case", "chunk", "ring_mib", "repeat", "target_tokens", "status", "prompt_n", "cache_n",
              "prompt_ms", "prefill_tps", "output_tokens", "decode_tps", "wall_s", "error"]
    base_url = f"http://127.0.0.1:{a.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def http(path, body=None, timeout=10):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=timeout) as response:
            return json.load(response)

    with open(out / "results.csv", "w", newline="", buffering=1) as cf:
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for index, (chunk, ring) in enumerate(pairs, 1):
            if port_busy(8081) or port_busy(a.port) or engines():
                raise RuntimeError("Another server/engine is running. Stopping sweep without touching it.")
            name = f"{index:02d}-chunk{chunk}-ring{ring}"
            case = out / name
            case.mkdir()
            test = copy.deepcopy(clean)
            test.update(port=a.port, host="127.0.0.1", log=str(case / "engine.log"), api_key="", mcp_servers={})
            test["env"] = dict(test.get("env", {}), DS4_CHUNK=str(chunk), DS4_RING_MB=str(ring))
            flag(test["args"], "--prefill-chunk", chunk)
            if a.tp is not None:
                flag(test["args"], "--tp", a.tp)
                flag(test["args"], "--threads", "on")
            if a.cpu_threads is not None:
                flag(test["args"], "--cpu-threads", a.cpu_threads)
            write_json(case / "config.json", test)
            env = dict(os.environ)
            # Remove ambient tuning knobs; config is the explicit source of engine settings.
            for key in list(env):
                if key.startswith(("DS4_", "MAYA_DS4_")):
                    del env[key]
            env.pop("STRATA_API_KEY", None)
            env.pop("MAYA_RESTART_ON_UPDATE", None)
            env.update(DS4_CHUNK=str(chunk), DS4_RING_MB=str(ring), PYTHONUNBUFFERED="1")
            print(f"[{index}/{len(pairs)}] {name}: loading", flush=True)
            with open(case / "server.log", "w") as log:
                proc = subprocess.Popen([str(root / ".venv/bin/python"), str(root / "serve/server.py"),
                    "--engine", "strata", "--config", str(case / "config.json"), "--host", "127.0.0.1",
                    "--port", str(a.port)], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                try:
                    start = time.monotonic()
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"Server exited with {proc.returncode}; inspect server.log")
                        try:
                            health = http("/health", timeout=2)
                            if health.get("status") == "ok" and not health.get("unloaded"):
                                break
                        except (OSError, ValueError):
                            pass
                        if time.monotonic() - start > a.startup_timeout:
                            raise TimeoutError("Engine startup timed out")
                        time.sleep(2)
                    write_json(case / "health.json", health)
                    # Same untimed long-prefill warm-up in each case, with a distinct prefix.
                    warm = {"model": cfg.get("model_name", "deepseek-v4.1-flash"),
                        "messages": [{"role": "user", "content": "Warm-up document. Summarize this text:\n" +
                            tok.decode(tokens[-min(8192, max(a.lengths)):])}],
                        "max_tokens": 64, "temperature": 0, "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False}}
                    write_json(case / "warmup.json", http("/v1/chat/completions", warm, a.request_timeout))
                    for rep, n, _ in prompts:
                        req = json.loads((pd / f"r{rep}-{n}.json").read_text())
                        started = time.monotonic()
                        response = http("/v1/chat/completions", req, a.request_timeout)
                        elapsed = time.monotonic() - started
                        write_json(case / f"response-r{rep}-{n}.json", response)
                        t = response.get("timings") or {}
                        if response.get("error") or not t.get("prompt_ms") or not t.get("prompt_n"):
                            raise RuntimeError("Missing valid per-request engine timings; see saved response")
                        reused = t.get("cache_n", 0)
                        status = "ok" if reused <= 0.01 * (t["prompt_n"] + reused) else "cached-exclude"
                        row = dict(case=name, chunk=chunk, ring_mib=ring, repeat=rep, target_tokens=n,
                            status=status, prompt_n=t["prompt_n"], cache_n=reused, prompt_ms=t["prompt_ms"],
                            prefill_tps=t.get("prompt_per_second"), output_tokens=t.get("predicted_n"),
                            decode_tps=t.get("predicted_per_second"), wall_s=round(elapsed, 3), error="")
                        writer.writerow(row)
                        cf.flush()
                        print(f"  r{rep} {n}: {row['prefill_tps']} prefill tok/s; {row['decode_tps']} decode; {status}", flush=True)
                    write_json(case / "completed.json", {"status": "complete"})
                except Exception as e:
                    write_json(case / "error.json", {"error": str(e)})
                    writer.writerow(dict(case=name, chunk=chunk, ring_mib=ring, status="failed", error=str(e)))
                    cf.flush()
                    print(f"  FAILED: {e}; continuing to next case", flush=True)
                finally:
                    stop_group(proc)
            time.sleep(3)
    with open(out / "results.csv") as f:
        rows = list(csv.DictReader(f))
    summary = []
    for case_name, length in sorted({(r["case"], r["target_tokens"]) for r in rows if r["status"] == "ok"}):
        group = [r for r in rows if r["status"] == "ok" and r["case"] == case_name and r["target_tokens"] == length]
        speeds = [float(r["prefill_tps"]) for r in group]
        decode = [float(r["decode_tps"]) for r in group if r["decode_tps"]]
        summary.append({"case": case_name, "target_tokens": length, "samples": len(group),
            "prefill_median": round(statistics.median(speeds), 2), "prefill_min": min(speeds),
            "prefill_max": max(speeds), "decode_median": round(statistics.median(decode), 2) if decode else ""})
    if summary:
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    print(f"Finished. Results and per-case medians: {out}. Failed/cached rows are excluded from summary.csv.", flush=True)
    failures = sum(r["status"] == "failed" for r in rows)
    print(f"Valid measurements: {sum(r['status'] == 'ok' for r in rows)}; failed cases: {failures}", flush=True)
    if not summary:
        sys.exit(1)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped; test process group cleaned up. Partial results retained.", file=sys.stderr)
        sys.exit(130)

