from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.metal.runtime import NativeModel

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL-MTP.gguf"
BANDWIDTH_BPS, FLOPS = 200e9, 13.6e12

def run(model, tokens, decode, batch, speculative=False, draft_tokens=2):
    sessions = [model.session() for _ in range(batch)]
    for session in sessions: session.configure(speculative=speculative, draft_tokens=draft_tokens)
    start = perf_counter()
    current = model.forward_batch(sessions, [tokens] * batch); ttft = perf_counter() - start
    start = perf_counter()
    for _ in range(decode):
        current = ([sessions[0].forward([current[0]])] if batch == 1 else model.forward_batch(sessions, current))
    elapsed, mapped = perf_counter() - start, sessions[0].mapped_bytes
    spec = [session.speculative_counters() for session in sessions]
    for session in sessions: session.close()
    return ttft, elapsed, mapped, sum(item["drafted_tokens"] for item in spec), sum(item["accepted_tokens"] for item in spec)


def main():
    p = argparse.ArgumentParser(description="Qwen3.5 native Metal 4 macro benchmark")
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=1024)
    p.add_argument("--batch", type=int, choices=range(1, 9), default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--draft-tokens", type=int, choices=range(1, 5), default=2)
    a = p.parse_args()
    print(f"[load] weights={a.weights}", flush=True); model = NativeModel(a.weights)
    params, model_bytes = model.parameter_count, model.weight_bytes
    rng = random.Random(1); tokens = [rng.randrange(model.vocab_size) for _ in range(a.prefill)]
    if a.speculative and not model.has_mtp: p.error("--speculative requires an MTP-enabled GGUF")
    print(f"# Qwen3.5 Metal 4 Benchmark\nweights={a.weights} dtype=float16 speculative={a.speculative} "
          f"draft_tokens={a.draft_tokens} "
          f"batch={a.batch} prefill={a.prefill} decode={a.decode} warmup={a.warmup} iters={a.iters}", flush=True)
    for i in range(a.warmup):
        print(f"warmup={i + 1}/{a.warmup}", flush=True)
        run(model, tokens, a.decode, a.batch, a.speculative, a.draft_tokens)
    ttft, decode, mapped, drafted, accepted = [], [], 0, 0, 0
    for i in range(a.iters):
        print(f"iter={i + 1}/{a.iters}", flush=True)
        ptime, dtime, mapped, proposed, kept = run(model, tokens, a.decode, a.batch, a.speculative, a.draft_tokens)
        ttft.append(ptime); decode.append(dtime); drafted += proposed; accepted += kept
    ttft_s, decode_s = sum(ttft) / len(ttft), sum(decode) / len(decode)
    tpot_s, ttl_s = decode_s / a.decode, ttft_s + decode_s
    throughput = a.batch * (a.prefill + a.decode) / ttl_s
    print(f"TTFT={ttft_s * 1000:.2f}ms TPOT={tpot_s * 1000:.2f}ms TTL={ttl_s:.2f}s "
          f"aggregate_tok/s={throughput:.2f} prefill_aggregate_tok/s={a.batch * a.prefill / ttft_s:.2f} "
          f"decode_aggregate_tok/s={a.batch / tpot_s:.2f} session_decode_tok/s={1 / tpot_s:.2f} "
          f"mapped_kv={mapped / 2**20:.2f}MiB "
          f"MBU={(model_bytes / tpot_s) / BANDWIDTH_BPS * 100:.2f}% "
          f"MFU={(2 * params * a.batch * a.prefill / ttft_s) / FLOPS * 100:.2f}% "
          f"drafted={drafted} accepted={accepted} acceptance={accepted / drafted * 100 if drafted else 0:.1f}%", flush=True)
    model.close()


if __name__ == "__main__": main()
