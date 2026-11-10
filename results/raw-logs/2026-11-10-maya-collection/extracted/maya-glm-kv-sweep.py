#!/usr/bin/env python3
"""GLM-5.3-Flash HIP prefill sweep, using Maya engine timings."""

import copy
import csv
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path("/opt/project-maya")
PORT = 8082
BASE = f"http://127.0.0.1:{PORT}"

CONFIGS = {
    "8k-chunks": ROOT / "maya-glm-5.3-flash-q4-2gpu-1m-reserve1g.json",
    "int8-kv": ROOT / "maya-glm-5.3-flash-q4-2gpu-1m-int8kv.json",
}

LENGTHS = (8192, 32768, 65536)
REPEATS = 2
OUTPUT_TOKENS = 128

OUT = (
    Path("/opt/maya-test-results")
    / ("glm-kv-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
)

sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def api(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({})
    )
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def port_busy():
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def wait_ready(proc, timeout=300):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited with status {proc.returncode}"
            )

        try:
            health = api("/health", timeout=3)
            if (
                health.get("status") == "ok"
                and not health.get("unloaded")
            ):
                return health
        except Exception:
            pass

        time.sleep(2)

    raise TimeoutError("Maya failed to become ready")


def generate_prompts(tokenizer):
    """Create identical document prompts for both configurations."""
    corpus_files = (
        [ROOT / "README.md"]
        + sorted((ROOT / "docs").glob("*.md"))
    )

    corpus = "\n\n".join(
        p.read_text(errors="replace")
        for p in corpus_files
        if p.is_file()
    )

    tokens = tokenizer.encode(corpus)
    if not tokens:
        raise RuntimeError("Empty benchmark corpus")

    while len(tokens) < max(LENGTHS) + 4096:
        tokens = tokens + tokens

    prompts = {}

    for repeat in range(REPEATS):
        for length in LENGTHS:
            # Different prefix prevents cross-request KV reuse.
            prefix = (
                f"Benchmark run {repeat}, document length {length}. "
                "Read the document and summarize its technical content.\n\n"
            )

            prefix_tokens = len(tokenizer.encode(prefix))
            document_tokens = length - prefix_tokens - 128

            offset = repeat * 997 + length // 16
            selected = tokens[offset:offset + document_tokens]

            if len(selected) != document_tokens:
                raise RuntimeError("Insufficient corpus tokens")

            prompt = prefix + tokenizer.decode(selected)

            prompts[(repeat, length)] = prompt

    return prompts


def effective_chunks(log):
    values = re.findall(
        r"glm prefill: CUDA\d+ chunks of (\d+) tokens",
        log
    )
    return [int(x) for x in values]


def main():
    if port_busy():
        raise RuntimeError(
            f"Port {PORT} is occupied. Stop the existing server first."
        )

    for config in CONFIGS.values():
        if not config.is_file():
            raise FileNotFoundError(config)

    OUT.mkdir(parents=True, exist_ok=False)
    print(f"Results: {OUT}", flush=True)

    source = json.loads(next(iter(CONFIGS.values())).read_text())
    tokenizer = Tokenizer.from_dir(Path(source["tokenizer"]))
    prompts = generate_prompts(tokenizer)

    for (repeat, length), prompt in prompts.items():
        (OUT / f"prompt-r{repeat}-{length}.txt").write_text(prompt)

    fields = [
        "configuration", "repeat", "target_tokens",
        "prompt_n", "cache_n", "prompt_ms",
        "prefill_tps", "output_tokens", "decode_tps",
        "wall_seconds", "effective_chunk", "status"
    ]

    with (OUT / "results.csv").open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for label, config_path in CONFIGS.items():
            if port_busy():
                raise RuntimeError(
                    "Port became occupied between configurations"
                )

            case = OUT / label
            case.mkdir()

            cfg = copy.deepcopy(json.loads(config_path.read_text()))

            # Preserve all inference settings.
            # Redirect only the engine log and server port.
            cfg["log"] = str(case / "engine.log")
            cfg["port"] = PORT
            cfg["host"] = "127.0.0.1"

            isolated = case / "config.json"
            save_json(isolated, cfg)

            print(f"\n=== {label} ===", flush=True)

            cmd = [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "serve/server.py"),
                "--engine", "strata",
                "--config", str(isolated),
                "--host", "127.0.0.1",
                "--port", str(PORT),
            ]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            with (case / "server.log").open("w") as logfile:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=logfile,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

                try:
                    health = wait_ready(proc)
                    save_json(case / "health.json", health)

                    engine_log = case / "engine.log"
                    log = (
                        engine_log.read_text(errors="replace")
                        if engine_log.exists() else ""
                    )

                    chunks = effective_chunks(log)
                    chunk = min(chunks) if chunks else None

                    print(
                        f"Effective chunk: {chunk or 'not reported'}",
                        flush=True
                    )

                    model = api("/v1/models")["data"][0]["id"]

                    # Untimed warm-up, as in the DeepSeek sweep.
                    warm = {
                        "model": model,
                        "messages": [{
                            "role": "user",
                            "content": "Summarize:\n"
                            + prompts[(0, 8192)][:16000],
                        }],
                        "temperature": 0,
                        "max_tokens": 64,
                        "stream": False,
                    }

                    save_json(
                        case / "warmup.json",
                        api("/v1/chat/completions", warm, timeout=600)
                    )

                    for repeat in range(REPEATS):
                        for length in LENGTHS:
                            prompt = prompts[(repeat, length)]

                            payload = {
                                "model": model,
                                "messages": [{
                                    "role": "user",
                                    "content": prompt,
                                }],
                                "temperature": 0,
                                "max_tokens": OUTPUT_TOKENS,
                                "stream": False,
                            }

                            # Measure exact chat token count.
                            expected = api(
                                "/api/tokens",
                                {"messages": payload["messages"]},
                                timeout=120,
                            )["tokens"]

                            started = time.monotonic()
                            response = api(
                                "/v1/chat/completions",
                                payload,
                                timeout=1800,
                            )
                            wall = time.monotonic() - started

                            save_json(
                                case / f"response-r{repeat}-{length}.json",
                                response,
                            )

                            timing = response.get("timings") or {}
                            prompt_n = timing.get("prompt_n")
                            prompt_ms = timing.get("prompt_ms")
                            cache_n = timing.get("cache_n", 0)

                            if (
                                not isinstance(prompt_n, (int, float))
                                or not isinstance(prompt_ms, (int, float))
                                or prompt_ms <= 0
                            ):
                                raise RuntimeError(
                                    "Maya did not return valid prompt timings; "
                                    f"inspect {case}"
                                )

                            tps = (
                                timing.get("prompt_per_second")
                                or prompt_n * 1000 / prompt_ms
                            )

                            status = (
                                "cached-exclude"
                                if cache_n > 0.01 * (prompt_n + cache_n)
                                else "ok"
                            )

                            row = {
                                "configuration": label,
                                "repeat": repeat,
                                "target_tokens": length,
                                "prompt_n": prompt_n,
                                "cache_n": cache_n,
                                "prompt_ms": round(prompt_ms, 2),
                                "prefill_tps": round(tps, 2),
                                "output_tokens": timing.get("predicted_n"),
                                "decode_tps": timing.get(
                                    "predicted_per_second"
                                ),
                                "wall_seconds": round(wall, 2),
                                "effective_chunk": chunk,
                                "status": status,
                            }

                            writer.writerow(row)
                            csvfile.flush()

                            print(
                                f"r{repeat} "
                                f"{length:6} target "
                                f"({expected} chat tokens): "
                                f"{tps:.1f} prefill tok/s; "
                                f"{row['decode_tps']} decode tok/s; "
                                f"{status}",
                                flush=True,
                            )

                except Exception:
                    print(
                        f"Failure in {label}. "
                        f"Inspect {case / 'server.log'} and "
                        f"{case / 'engine.log'}",
                        flush=True,
                    )
                    raise
                finally:
                    stop(proc)

            time.sleep(3)

    print(f"\nCompleted. Results: {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
