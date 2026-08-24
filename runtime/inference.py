from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ.setdefault("USE_TORCH", "0")
from transformers import AutoTokenizer

from backend.metal.runtime import NativeModel


class _ForwardRequest:
    __slots__ = "session", "tokens", "params", "event", "result", "error"

    def __init__(self, session, tokens, params):
        self.session, self.tokens, self.params = session, tokens, params
        self.event, self.result, self.error = threading.Event(), None, None


class _Scheduler:
    """Continuously rebuilds one mixed target batch after every scheduler step."""

    def __init__(self, native, max_batch=8):
        self.native, self.max_batch = native, max_batch
        self.condition, self.incoming, self.requests, self.closed = threading.Condition(), [], [], False
        self.worker = threading.Thread(target=self._run, name="infeng-scheduler", daemon=True); self.worker.start()

    def forward(self, session, tokens, temperature, top_p, top_k):
        params = (temperature, top_p, top_k)
        with self.condition:
            if self.closed: raise RuntimeError("inference engine is closed")
            request = _ForwardRequest(session, tokens, params); self.incoming.append(request); self.condition.notify()
        request.event.wait()
        if request.error: raise request.error
        return request.result

    def _run(self):
        while True:
            with self.condition:
                while not self.requests and not self.incoming and not self.closed: self.condition.wait()
                self.requests[0:0] = self.incoming; self.incoming.clear()
                if self.closed:
                    for item in self.requests: item.error = RuntimeError("inference engine is closed"); item.event.set()
                    self.requests.clear(); return
                params = self.requests[0].params
                batch = [item for item in self.requests if item.params == params][:self.max_batch]
            try:
                outputs, ready = self.native.forward_step([item.session for item in batch],
                                                          [item.tokens for item in batch], *params)
                for item, output, done in zip(batch, outputs, ready):
                    if done: item.result = output; item.event.set()
            except Exception as exc:
                for item in batch: item.error = exc; item.event.set()
                ready = [True] * len(batch)
            with self.condition:
                untouched = [item for item in self.requests if item not in batch]
                self.requests = untouched + [item for item, done in zip(batch, ready) if not done]

    def close(self):
        with self.condition:
            if self.closed: return
            self.closed = True; self.condition.notify_all()
        self.worker.join()


class Session:
    def __init__(self, engine: InferenceEngine):
        self.engine, self.native = engine, None
        self.pending, self.closed, self.sealed = [], False, True
        self.native = engine.native.session()

    def _tokens(self, message, thinking):
        text = self.engine.tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=thinking)
        prefix = "" if not self.pending else ("\n" if self.sealed else "<|im_end|>\n")
        return list(self.engine.tokenizer.encode(prefix + text, add_special_tokens=False))

    def generate(self, message: str, max_new_tokens: int | None = None, thinking: bool = False,
                 stop_token_ids: list[int] | None = None, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int | None = None, speculative: bool = False, draft_tokens: int = 2):
        if self.closed: raise RuntimeError("session is closed")
        pending = self.pending + self._tokens(message, thinking); self.pending = pending
        remaining = self.engine.max_context - self.native.length - len(pending)
        count = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
        if count < 0: raise ValueError(f"context exceeds {self.engine.max_context} tokens")
        im_end = self.engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if stop_token_ids is None: stop_token_ids = [x for x in (self.engine.tokenizer.eos_token_id, im_end) if x is not None]
        stopped = False
        try:
            for step in range(count):
                remaining_tokens = count - step
                active_speculation = speculative and remaining_tokens > 1
                active_drafts = min(draft_tokens, max(1, remaining_tokens - 1))
                self.native.configure(speculative=active_speculation, draft_tokens=active_drafts,
                                      stop_token_ids=stop_token_ids)
                query = pending if step == 0 else pending[0]
                token = self.engine._scheduler.forward(self.native, query, temperature, top_p, top_k)
                self.pending = pending = [token]
                if token in stop_token_ids: self.sealed, stopped = token == im_end, True; break
                yield token
        finally:
            if not stopped: self.sealed = False

    @property
    def mapped_kv_bytes(self): return self.native.mapped_bytes

    @property
    def sequence_id(self): return self.native.sequence_id

    def close(self):
        if not self.closed and self.native: self.native.close()
        self.closed = True

    def __del__(self): self.close()


class InferenceEngine:
    def __init__(self, weights: str | Path, tokenizer: str, *, max_context: int = 65536):
        if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
        self.native, self._scheduler = None, None
        self.max_context, self.device, self.dtype = max_context, "metal4", "float16"
        self.native = NativeModel(weights, max_context=max_context)
        self._scheduler = _Scheduler(self.native)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def session(self):
        return Session(self)

    def close(self):
        if getattr(self, "_scheduler", None): self._scheduler.close(); self._scheduler = None
        if self.native: self.native.close(); self.native = None

    def __del__(self): self.close()
