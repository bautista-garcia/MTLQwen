from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import InferenceEngine

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
MTP_WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL-MTP.gguf"
DFLASH_WEIGHTS = ROOT / "weights/qwen35-9b-dflash-Q4_K_M.gguf"
DFLASH_MATH_PROMPT = [
  248045, 846, 198, 4199, 1599, 6572, 4220, 25030, 3319, 39852, 1503, 220, 16, 24, 21, 599, 30, 248046, 198, 248045, 74455, 198, 248068, 198
]
FIRST_TURN_RESPONSE = ("Hello! It seems like your message got cut off. Could you please share your "
                       "name or let me know how I can assist you today? 😊")
NATIVE_GREEDY_MULTITURN = ("Nice to meet you, Bautista! How can I help you today? Whether you have questions, need assistance with a task, "
                           "or just want to chat, feel free to ask! 😊")
LONG_PREFILL_PROMPT = "How much money do I need to save up in order to retire at 30, with investments included?"


def complete(sequence, prompt=(), limit=None):
  cursor, output = sequence.append(prompt), []
  while limit is None or len(output) < limit:
    token = sequence.read(cursor)
    if token is None: break
    cursor += 1
    if token in sequence.stop_token_ids: return output
    output.append(token)
  if limit is not None:
    sequence.cancel()
    while sequence.read(cursor) is not None:
      cursor += 1
  return output


def one(sequence, prompt=()):
  return complete(sequence, prompt, 1)[0]


def turn(tokenizer, message, thinking=False, prefix=""):
  prompt = tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
  return tokenizer.encode(prefix + prompt, add_special_tokens=False)


def parallel(calls):
  with ThreadPoolExecutor(max_workers=len(calls)) as pool:
    return list(pool.map(lambda call: call(), calls))


def test_greedy_multiturn_generation():
  os.environ.setdefault("USE_TORCH", "0")
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-9B"))
  im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
  stop_ids = [token for token in (tokenizer.eos_token_id, im_end) if token is not None]
  engine = InferenceEngine(WEIGHTS)
  assert one(engine.sequence(stop_token_ids=stop_ids), turn(tokenizer, LONG_PREFILL_PROMPT)) == 11678
  sequence = engine.sequence(stop_token_ids=stop_ids)
  first_ids = complete(sequence, turn(tokenizer, "Hello, my name is"))
  first = tokenizer.decode(first_ids, skip_special_tokens=False)
  assert first.strip() == FIRST_TURN_RESPONSE, repr(first)
  prefix = "\n" if first_ids[-1] == im_end else "<|im_end|>\n"
  ours = tokenizer.decode(complete(sequence, turn(tokenizer, "Bautista", prefix=prefix)), skip_special_tokens=False)
  sequence.close()
  engine.close()
  assert ours.strip() == NATIVE_GREEDY_MULTITURN, repr(ours)


def test_continuous_mixed_batch_and_prefix_restore():
  model = InferenceEngine(WEIGHTS, draft_weights=MTP_WEIGHTS)
  assert model.drafter == "mtp"
  reference = [model.sequence(), model.sequence(), model.sequence(speculative=True)]
  actual = [model.sequence(), model.sequence(), model.sequence(speculative=True)]
  extra = []
  try:
    prompt = list(range(1, 201))
    expected_pp = complete(reference[0], prompt, 1)
    one(reference[1], [10, 11, 12])
    expected_tg = complete(reference[1], limit=1)
    one(reference[2], [20, 21, 22])
    expected_spec = complete(reference[2], limit=1)
    one(actual[1], [10, 11, 12])
    one(actual[2], [20, 21, 22])
    observed = parallel([
      lambda: complete(actual[0], prompt, 1),
      lambda: complete(actual[1], limit=1),
      lambda: complete(actual[2], limit=1),
    ])
    assert observed == [expected_pp, expected_tg, expected_spec]
    assert actual[0].length == 200

    for sequence in reference + actual:
      sequence.close()
    checkpoint = model.sequence(speculative=True)
    restored = model.sequence(speculative=True)
    extra += [checkpoint, restored]
    checkpoint_prompt = list(range(1, 521))
    expected = one(checkpoint, checkpoint_prompt)
    assert checkpoint.length == 520
    assert one(restored, checkpoint_prompt) == expected and restored.length == 520
    assert complete(checkpoint, limit=4) == complete(restored, limit=4)

    stop_reference = model.sequence(speculative=True)
    extra.append(stop_reference)
    anchor = one(stop_reference, DFLASH_MATH_PROMPT)
    stop = one(stop_reference)
    stop_sequence = model.sequence(stop_token_ids=[stop], speculative=True)
    extra.append(stop_sequence)
    assert one(stop_sequence, DFLASH_MATH_PROMPT) == anchor
    assert complete(stop_sequence) == []
    assert stop_sequence.length == len(DFLASH_MATH_PROMPT) + 2
    assert stop_sequence.speculative_counters()["accepted_tokens"] == 1

    edge_ref, draft_ref = model.sequence(speculative=True), model.sequence(speculative=True)
    edge_prompt, draft_prompt = list(range(1, 512)), [50, 51, 52]
    edge_anchor, draft_anchor = one(edge_ref, edge_prompt), one(draft_ref, draft_prompt)
    expected_edge = complete(edge_ref, limit=1)
    expected_draft = complete(draft_ref, limit=4)
    edge_ref.close()
    draft_ref.close()
    edge, draft_sequence = model.sequence(speculative=True), model.sequence(speculative=True)
    extra += [edge, draft_sequence]
    assert one(edge, edge_prompt) == edge_anchor and one(draft_sequence, draft_prompt) == draft_anchor
    observed = parallel([
      lambda: complete(edge, limit=1),
      lambda: complete(draft_sequence, limit=4),
    ])
    assert observed == [expected_edge, expected_draft] and edge.length == 512

    rollback_ref, rollback = model.sequence(speculative=True), model.sequence(speculative=True)
    extra += [rollback_ref, rollback]
    anchor = one(rollback_ref, [30, 31, 32])
    assert one(rollback, [30, 31, 32]) == anchor
    assert one(rollback) == one(rollback_ref)
  finally:
    for sequence in reference + actual + extra:
      sequence.close()
    model.close()


@pytest.mark.skipif(not DFLASH_WEIGHTS.exists(), reason="reference DFlash GGUF is unavailable")
def test_dflash_greedy_contract_and_configuration():
  model = InferenceEngine(WEIGHTS, draft_weights=DFLASH_WEIGHTS, max_context=1024)
  assert model.drafter == "dflash"
  reference, actual = model.sequence(), model.sequence(speculative=True)
  extra = []
  try:
    expected = complete(reference, DFLASH_MATH_PROMPT, 64)
    observed = complete(actual, DFLASH_MATH_PROMPT, 64)
    assert observed == expected
    counters = actual.speculative_counters()
    assert counters["drafted_tokens"] and counters["acceptance_rate"] >= 0.45
    assert model.mapped_bytes == 56 << 20

    checkpoint, restored = model.sequence(speculative=True), model.sequence(speculative=True)
    extra += [checkpoint, restored]
    prompt = list(range(1, 521))
    value = one(checkpoint, prompt)
    assert one(restored, prompt) == value
    assert complete(checkpoint, limit=8) == complete(restored, limit=8)
  finally:
    reference.close()
    actual.close()
    for sequence in extra:
      sequence.close()
    model.close()


def test_invalid_drafter_weights():
  with pytest.raises(RuntimeError):
    InferenceEngine(WEIGHTS, draft_weights=WEIGHTS, max_context=128)


@pytest.mark.parametrize("draft_weights", [None, MTP_WEIGHTS, DFLASH_WEIGHTS], ids=["target", "mtp", "dflash"])
def test_mixed_batch_sampling_matches_greedy(draft_weights):
  if draft_weights is not None and not draft_weights.exists(): pytest.skip("drafter GGUF is unavailable")
  model = InferenceEngine(WEIGHTS, draft_weights=draft_weights, max_context=1024)
  sequences = [model.sequence()]
  try:
    expected = complete(sequences[0], DFLASH_MATH_PROMPT, 12)
    sequences[0].close()
    # Top-k=1 exercises sampling and its packed RNG outputs with deterministic tokens.
    sequences = [model.sequence(temperature=1.0, top_k=1, speculative=bool(row % 2)) for row in range(8)]
    observed = parallel([lambda sequence=sequence: complete(sequence, DFLASH_MATH_PROMPT, 12) for sequence in sequences])
    assert observed == [expected] * len(sequences)
    if draft_weights is not None:
      assert all(sequence.speculative_counters()["drafted_tokens"] for sequence in sequences[1::2])
  finally:
    for sequence in sequences:
      sequence.close()
    model.close()


def test_allocator_failure_is_atomic():
  model = InferenceEngine(WEIGHTS, max_context=16)
  owner, waiting = model.sequence(), model.sequence()
  try:
    one(owner, list(range(1, 17)))
    with pytest.raises(RuntimeError, match="no evictable capacity"):
      one(waiting, [42])
    assert owner.length == 16 and waiting.length == 0
    owner.close()
    assert one(waiting) >= 0
  finally:
    owner.close()
    waiting.close()
    model.close()


def test_prefix_cache_sparse_aliases():
  model = InferenceEngine(WEIGHTS, max_context=4096)
  source = model.sequence()
  aliases = [model.sequence() for _ in range(4)]
  prompt = list(range(1, 521))
  try:
    expected = one(source, prompt)
    mapped = model.mapped_bytes
    assert parallel([lambda sequence=sequence: one(sequence, prompt) for sequence in aliases]) == [expected] * len(aliases)
    assert all(sequence.length == 520 for sequence in aliases) and model.mapped_bytes == mapped
    source.close()
    for sequence in aliases:
      sequence.close()
    reused = model.sequence()
    try:
      assert one(reused, prompt) == expected and reused.length >= 520
    finally:
      reused.close()
  finally:
    source.close()
    for sequence in aliases:
      sequence.close()
    model.close()


def test_sparse_pool_and_all_slots():
  model = InferenceEngine(WEIGHTS, max_context=1024)
  prompts = [[row * 32 + token + 1 for token in range(16)] for row in range(8)]
  references = []
  for prompt in prompts:
    sequence = model.sequence()
    references.append(one(sequence, prompt))
    sequence.close()
  sequences = [model.sequence() for _ in range(8)]
  try:
    assert parallel([lambda row=row: one(sequences[row], prompts[row]) for row in range(8)]) == references
    assert model.mapped_bytes == 32 << 20
    with pytest.raises(RuntimeError, match="maximum live sequence count"):
      model.sequence()
  finally:
    for sequence in sequences:
      sequence.close()
    model.close()
