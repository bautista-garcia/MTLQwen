from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.inference import InferenceEngine, _Scheduler
from backend.metal.runtime import NativeModel

WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL.gguf"
MTP_WEIGHTS = ROOT / "weights/Qwen3.5-9B-UD-Q4_K_XL-MTP.gguf"
FIRST_TURN_RESPONSE = (
    "Hello! It seems like your message got cut off. Could you please share your "
    "name or let me know how I can assist you today? \U0001f60a"
)
# This intentionally differs from llama.cpp: the append-only session preserves the exact first-turn prompt tokens in
# its KV cache, including Qwen's empty thinking block, while llama.cpp rerenders history without that historical block.
NATIVE_GREEDY_MULTITURN = (
    "Nice to meet you, Bautista! How can I help you today? Whether you have questions, need assistance with a task, "
    "or just want to chat, feel free to ask! \U0001f60a"
)

def test_greedy_multiturn_generation():
    engine = InferenceEngine(WEIGHTS, os.getenv("QWEN35_TOKENIZER", "Qwen/Qwen3.5-9B"))
    session = engine.session()
    first = engine.tokenizer.decode(list(session.generate("Hello, my name is", max_new_tokens=64, thinking=False,
                                                          temperature=0.0)), skip_special_tokens=False)
    assert first.strip() == FIRST_TURN_RESPONSE, repr(first)
    ours = engine.tokenizer.decode(list(session.generate("Bautista", max_new_tokens=64, thinking=False,
                                                          temperature=0.0)), skip_special_tokens=False)
    engine.close()
    assert ours.strip() == NATIVE_GREEDY_MULTITURN, repr(ours)


def test_mixed_prefill_target_and_speculative_batch():
    model = NativeModel(MTP_WEIGHTS); reference = [model.session() for _ in range(3)]
    actual = [model.session() for _ in range(3)]; reference[2].configure(speculative=True)
    actual[2].configure(speculative=True)
    extra = []
    try:
        prompt = list(range(1, 201)); expected_pp = reference[0].forward(prompt)
        target_anchor = reference[1].forward([10, 11, 12]); expected_tg = reference[1].forward([target_anchor])
        spec_anchor = reference[2].forward([20, 21, 22]); expected_spec = reference[2].forward([spec_anchor])
        actual_target = actual[1].forward([10, 11, 12]); actual_spec = actual[2].forward([20, 21, 22])
        outputs, ready = model.forward_step(actual, [prompt, [actual_target], [actual_spec]])
        assert ready == [0, 1, 1] and outputs[1:] == [expected_tg, expected_spec]
        assert [session.length for session in actual] == [124, 4, 4]
        assert actual[0].forward(prompt) == expected_pp and actual[0].length == 200

        checkpoint = model.session(); extra.append(checkpoint); checkpoint_prompt = list(range(1, 521)); lengths = []
        while True:
            checkpoint_output, done = model.forward_step([checkpoint], [checkpoint_prompt]); lengths.append(checkpoint.length)
            if done[0]: break
        assert lengths == [128, 256, 384, 512, 520]
        restored = model.session(); extra.append(restored)
        assert restored.forward(checkpoint_prompt) == checkpoint_output[0]
        checkpoint.configure(speculative=True); restored.configure(speculative=True)
        assert checkpoint.forward(checkpoint_output) == restored.forward(checkpoint_output)

        for session in reference: session.close()
        rollback_ref, rollback = model.session(), model.session(); extra.extend([rollback_ref, rollback])
        anchor = rollback_ref.forward([30, 31, 32]); assert rollback.forward([30, 31, 32]) == anchor
        rollback.configure(speculative=True); before = rollback.length
        with pytest.raises(RuntimeError, match="top_k"):
            model.forward_step([rollback], [[anchor]], temperature=1.0, top_k=-1)
        assert rollback.length == before and rollback.pending_outputs == 0
        assert rollback.forward([anchor]) == rollback_ref.forward([anchor])
    finally:
        for session in reference + actual + extra: session.close()
        model.close()


def test_scheduler_rebatches_prefill_with_new_decode():
    class FakeNative:
        def __init__(self): self.calls, self.entered, self.release = [], threading.Event(), threading.Event()

        def forward_step(self, sessions, tokens, *_):
            self.calls.append((sessions, tokens))
            if len(self.calls) == 1:
                self.entered.set(); assert self.release.wait(2); return [0], [0]
            return [101, 202], [1, 1]

    native = FakeNative(); scheduler = _Scheduler(native); results = {}
    prefill = threading.Thread(target=lambda: results.setdefault("pp", scheduler.forward("pp", [1] * 200, 0, 1, None)))
    decode = threading.Thread(target=lambda: results.setdefault("tg", scheduler.forward("tg", 7, 0, 1, None)))
    try:
        prefill.start(); assert native.entered.wait(2); decode.start(); native.release.set()
        prefill.join(2); decode.join(2)
        assert set(native.calls[1][0]) == {"pp", "tg"}
        assert results == dict(zip(native.calls[1][0], [101, 202]))
    finally:
        native.release.set(); scheduler.close(); prefill.join(2); decode.join(2)


def test_allocator_failure_is_atomic():
    model = NativeModel(WEIGHTS, max_context=16); owner, waiting = model.session(), model.session()
    try:
        assert owner.mapped_bytes == waiting.mapped_bytes == 0
        owner.forward(list(range(1, 17)))
        assert owner.mapped_bytes > 0
        with pytest.raises(RuntimeError, match="no evictable capacity"): waiting.forward([42])
        assert owner.length == 16 and waiting.length == 0
        owner.close()
        waiting.forward([42]); assert waiting.length == 1
    finally:
        owner.close(); waiting.close(); model.close()


def test_prefix_cache_sparse_aliases():
    model = NativeModel(WEIGHTS, max_context=4096); source = model.session(); aliases = [model.session() for _ in range(4)]
    prompt = list(range(1, 521))
    try:
        expected = source.forward(prompt); mapped = source.mapped_bytes
        outputs, ready = model.forward_step(aliases, [prompt] * len(aliases))
        assert ready == [1] * len(aliases) and outputs == [expected] * len(aliases)
        assert all(session.length == 520 for session in aliases) and source.mapped_bytes == mapped
        source.close()
        for session in aliases: session.close()
        reused = model.session()
        try: assert reused.forward(prompt) == expected and reused.length == 520
        finally: reused.close()
    finally:
        source.close()
        for session in aliases: session.close()
        model.close()


def test_sparse_pool_eviction_and_all_slots():
    model = NativeModel(WEIGHTS, max_context=1024)
    prompts = [[row * 32 + token + 1 for token in range(16)] for row in range(8)]; expected = []
    try:
        for prompt in prompts:
            reference = model.session()
            try: expected.append(reference.forward(prompt))
            finally: reference.close()
        sessions = [model.session() for _ in range(8)]
        try:
            assert model.forward_batch(sessions, prompts) == expected
            assert len({session.sequence_id for session in sessions}) == 8
            assert sessions[0].mapped_bytes == 32 << 20
            with pytest.raises(RuntimeError, match="maximum live sequence count"): model.session()
        finally:
            for session in sessions: session.close()
    finally:
        model.close()
