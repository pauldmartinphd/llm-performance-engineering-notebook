#!/usr/bin/env python3

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/opt/project-maya")
OUT = ROOT / "bench-glm-long-results"
OUT.mkdir(exist_ok=True)

PORT = 8082
BASE = f"http://127.0.0.1:{PORT}"

CONFIGS = {
    "8k-chunks": "maya-glm-5.3-flash-q4-2gpu-1m-reserve1g.json",
    "16k-chunks": "maya-glm-5.3-flash-q4-2gpu-1m-prefill32k-8gb.json",
}

LENGTHS = (32768, 65536)
MAX_NEW_TOKENS = 256

PARAGRAPH = (
    "A computer system processes data through several stages. "
    "Memory bandwidth, latency, caching, instruction throughput, "
    "and parallel execution determine overall performance. "
    "Measurements distinguish initialization, prefill, and decoding. "
    "The correctness of each computation must remain unchanged. "
)

def api(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def messages_for(text):
    return [{"role": "user", "content": text}]

def token_count(text):
    return api("/api/tokens", {"messages": messages_for(text)})["tokens"]

def make_prompt(target):
    # Binary-search character length using Maya's actual tokenizer.
    # Leave a small margin for tokenization boundary effects.
    wanted = target - 8
    text = PARAGRAPH * ((target // 30) + 100)
    low, high = 0, len(text)

    while low < high:
        mid = (low + high + 1) // 2
        n = token_count(text[:mid])
        if n <= wanted:
            low = mid
        else:
            high = mid - 1

    result = text[:low]
    count = token_count(result)
    print(f"Target {target}: actual {count} prompt tokens", flush=True)
    return result, count

def wait_ready(proc):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Maya exited with code {proc.returncode}")
        try:
            api("/health", timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Maya did not become ready")

def benchmark(model, prompt, prompt_tokens):
    payload = {
        "model": model,
        "messages": messages_for(prompt),
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "stream": True,
    }

    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    start = time.monotonic()
    first = None
    pieces = []
    finish_reason = None

    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue

            data = line[6:]
            if data == "[DONE]":
                break

            event = json.loads(data)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")

                if content:
                    if first is None:
                        first = time.monotonic()
                    pieces.append(content)

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    end = time.monotonic()
    if first is None:
        raise RuntimeError("No output received")

    generated = "".join(pieces)
    output_tokens = api(
        "/tokenize", {"content": generated}
    )["count"]

    return {
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": round(first - start, 3),
        "total_seconds": round(end - start, 3),
        "output_tokens_estimate": output_tokens,
        "decode_tps_estimate": round(
            max(0, output_tokens - 1) / (end - first), 2
        ) if end > first else None,
        "finish_reason": finish_reason,
    }

results = []
prompts = {}

try:
    for label, config in CONFIGS.items():
        print(f"\n=== {label} ===", flush=True)

        with (OUT / f"{label}.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    "./maya.sh",
                    "--backend", "hip",
                    "--config", config,
                    "--port", str(PORT),
                    "--plain",
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            try:
                wait_ready(proc)

                model = api("/v1/models")["data"][0]["id"]

                for length in LENGTHS:
                    if length not in prompts:
                        prompts[length] = make_prompt(length)
                        (OUT / f"prompt-{length}.txt").write_text(
                            prompts[length][0]
                        )

                    prompt, count = prompts[length]
                    print(
                        f"{label}: running {count} tokens",
                        flush=True
                    )

                    result = benchmark(model, prompt, count)
                    result["configuration"] = label
                    results.append(result)

                    print(json.dumps(result, indent=2), flush=True)

                    (OUT / "results.json").write_text(
                        json.dumps(results, indent=2) + "\n"
                    )

            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

finally:
    if results:
        (OUT / "results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )

print("\n=== RESULTS ===")
for r in results:
    print(
        f"{r['configuration']:12} "
        f"{r['prompt_tokens']:6} tokens  "
        f"TTFT={r['ttft_seconds']:.2f}s  "
        f"decode~={r['decode_tps_estimate']} tok/s"
    )
