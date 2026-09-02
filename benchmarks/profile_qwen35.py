from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.inference import InferenceEngine

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
MATH_PROMPT = [
  248045, 846, 198, 4199, 1599, 6572, 4220, 25030, 3319, 39852, 1503, 220, 16, 24, 21, 599, 30, 248046, 198, 248045, 74455, 198, 248068, 198
]


def snapshot(model):
  return {"counters": model.counters(), "kernels": {(counter["phase"], counter["name"]): counter for counter in model.kernel_counters()}}


def delta(before, after):
  kernels = []
  for key, current in after["kernels"].items():
    previous = before["kernels"].get(key, {"gpu_time_ns": 0, "launches": 0})
    launches = current["launches"] - previous["launches"]
    if launches:
      kernels.append({
        "phase": current["phase"],
        "name": current["name"],
        "gpu_time_ns": current["gpu_time_ns"] - previous["gpu_time_ns"],
        "launches": launches
      })
  return {
    "gpu_time_ns": after["counters"]["gpu_time_ns"] - before["counters"]["gpu_time_ns"],
    "passes": after["counters"]["passes"] - before["counters"]["passes"],
    "kernels": kernels
  }


def merge(target, samples):
  for sample in samples:
    key = sample["phase"], sample["name"]
    current = target.setdefault(key, {"phase": sample["phase"], "name": sample["name"], "gpu_time_ns": 0, "launches": 0})
    current["gpu_time_ns"] += sample["gpu_time_ns"]
    current["launches"] += sample["launches"]


def report(label, totals, gpu_time_ns, forwards, tokens_per_forward):
  rows = sorted(totals.values(), key=lambda sample: sample["gpu_time_ns"], reverse=True)
  whole_ms = gpu_time_ns / forwards / 1e6
  baseline_tps = tokens_per_forward / (whole_ms / 1000)
  covered_ms = sum(sample["gpu_time_ns"] for sample in rows) / forwards / 1e6
  print(f"\n[{label}] complete command-buffer GPU time={whole_ms:.3f} ms/forward, GPU throughput={baseline_tps:.2f} tok/s")
  print("kernel                                      launches/forward  ms/forward  % of forward   "
        "+tok/s @1.2x   +tok/s @1.5x     +tok/s @2x     +tok/s @4x")
  for sample in rows:
    milliseconds = sample["gpu_time_ns"] / forwards / 1e6
    fraction = milliseconds / whole_ms if whole_ms else 0
    gains = [baseline_tps * (1 / (1 - fraction + fraction / speedup) - 1) for speedup in (1.2, 1.5, 2, 4)]
    name = f"{sample['phase']}/{sample['name']}"
    print(f"{name:<44} {sample['launches'] / forwards:>16.2f} {milliseconds:>11.3f} "
          f"{fraction * 100:>12.2f}% {gains[0]:>14.2f} {gains[1]:>14.2f} {gains[2]:>14.2f} {gains[3]:>14.2f}")
  if covered_ms < whole_ms:
    fraction = (whole_ms - covered_ms) / whole_ms
    gains = [baseline_tps * (1 / (1 - fraction + fraction / speedup) - 1) for speedup in (1.2, 1.5, 2, 4)]
    print(f"{'unattributed gaps/barriers':<44} {'':>16} {whole_ms - covered_ms:>11.3f} "
          f"{fraction * 100:>12.2f}% {gains[0]:>14.2f} {gains[1]:>14.2f} {gains[2]:>14.2f} {gains[3]:>14.2f}")


def main():
  parser = argparse.ArgumentParser(description="Profile Qwen3.5 kernels within complete Metal forward passes")
  parser.add_argument("--weights", type=Path, default=WEIGHTS)
  parser.add_argument("--drafter", choices=("none", "mtp", "dflash"), default="none")
  parser.add_argument("--draft-weights", type=Path)
  parser.add_argument("--max-context", type=int, default=4096)
  parser.add_argument("--speculative", action="store_true")
  parser.add_argument("--draft-tokens", type=int)
  parser.add_argument("--prefill", type=int, default=128)
  parser.add_argument("--prompt-ids", help="comma-separated token IDs; overrides --prefill random input")
  parser.add_argument("--prompt-preset", choices=("math", ), help="use a deterministic chat-formatted prompt")
  parser.add_argument("--decode", type=int, default=32)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iters", type=int, default=5)
  args = parser.parse_args()
  if args.prompt_ids and args.prompt_preset: parser.error("--prompt-ids and --prompt-preset are mutually exclusive")
  model = InferenceEngine(args.weights, drafter=args.drafter, draft_weights=args.draft_weights, profile=True, max_context=args.max_context)
  rng = random.Random(1)
  tokens = ([int(token) for token in args.prompt_ids.split(",")]
            if args.prompt_ids else MATH_PROMPT if args.prompt_preset else [rng.randrange(model.vocab_size) for _ in range(args.prefill)])
  args.prefill = len(tokens)
  for _ in range(args.warmup):
    session = model.session()
    session.configure(speculative=args.speculative, draft_tokens=args.draft_tokens)
    token = session.forward(tokens)
    for _ in range(args.decode):
      token = session.forward([token])
    session.close()
  prefill, decode, prefill_gpu, decode_gpu = {}, {}, 0, 0
  for _ in range(args.iters):
    session = model.session()
    session.configure(speculative=args.speculative, draft_tokens=args.draft_tokens)
    before = snapshot(model)
    token = session.forward(tokens)
    middle = snapshot(model)
    prefill_delta = delta(before, middle)
    for _ in range(args.decode):
      token = session.forward([token])
    after = snapshot(model)
    decode_delta = delta(middle, after)
    session.close()
    merge(prefill, prefill_delta["kernels"])
    merge(decode, decode_delta["kernels"])
    prefill_gpu += prefill_delta["gpu_time_ns"]
    decode_gpu += decode_delta["gpu_time_ns"]
  report("prefill", prefill, prefill_gpu, args.iters, args.prefill)
  report("decode", decode, decode_gpu, args.iters * args.decode, 1)
  model.close()


if __name__ == "__main__": main()
