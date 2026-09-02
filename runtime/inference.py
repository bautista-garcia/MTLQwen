from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import weakref
from concurrent.futures import Future
from itertools import accumulate
from pathlib import Path

ROOT, MAX_BATCH, MAX_DRAFTS, BLOCK, VOCAB = Path(__file__).resolve().parents[1], 8, 7, 128, 248320
LIB, KERNELS = ROOT / ".build/libinfeng.dylib", ROOT / "backend/metal/kernel"
SOURCES = [ROOT / path for path in ("backend/metal/device.cpp", "model/qwen35_weights.cpp", "model/qwen_ops.cpp", "model/qwen.cpp")]


class _Info(ctypes.Structure):
  _fields_ = [(name, ctypes.c_uint64) for name in ("parameters", "weight_bytes", "mapped_bytes", "gpu_time_ns", "passes")]


def _load():
  headers = [*ROOT.glob("backend/metal/*.hpp"), *ROOT.glob("model/*.hpp"), *(ROOT / "third_party/metal-cpp").rglob("*.hpp")]
  if not LIB.exists() or any(path.stat().st_mtime > LIB.stat().st_mtime for path in [*SOURCES, *headers]):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    ggml = subprocess.check_output(["pkg-config", "--cflags", "--libs", "ggml"], text=True).split()
    subprocess.run(["xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-fblocks", "-fvisibility=hidden", "-DMETALCPP_SYMBOL_VISIBILITY_HIDDEN",
                    "-dynamiclib", *(str(path) for path in SOURCES), "-I", str(ROOT / "third_party/metal-cpp"), "-I", str(ROOT), *ggml, "-lggml-base",
                    "-framework", "Foundation", "-framework", "Metal", "-o", str(LIB)], check=True)
  lib = ctypes.CDLL(LIB)
  void, voids, ints, uints, bytes_ = ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint8)
  lib.infeng_last_error.restype = ctypes.c_char_p
  lib.infeng_model_create.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32)
  lib.infeng_model_create.restype = void
  lib.infeng_model_release.argtypes = (void, )
  lib.infeng_sequence_create.argtypes, lib.infeng_sequence_create.restype = (void, ), void
  lib.infeng_sequence_release.argtypes = (void, )
  lib.infeng_prefix.argtypes = (void, ints, ctypes.c_uint32, uints)
  lib.infeng_forward.argtypes = (void, voids, ints, uints, uints, uints, bytes_, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float,
                                 ctypes.c_int32, ctypes.POINTER(ints))
  lib.infeng_commit.argtypes = (void, ints, uints, uints, uints, bytes_)
  lib.infeng_abort.argtypes = (void, )
  lib.infeng_info.argtypes = (void, ctypes.POINTER(_Info))
  lib.infeng_kernel_counter_count.argtypes, lib.infeng_kernel_counter_count.restype = (void, ), ctypes.c_uint32
  lib.infeng_kernel_counter.argtypes = (void, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
                                        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64))
  return lib


_LIB = _load()


def _error():
  return (_LIB.infeng_last_error() or b"native Metal operation failed").decode()


def _check(value):
  if not value: raise RuntimeError(_error())
  return value


def _status(value):
  if value: raise RuntimeError(_error())


class _Scheduler:

  def __init__(self, engine):
    self.engine, self.condition, self.requests, self.closed = engine, threading.Condition(), [], False
    self.worker = threading.Thread(target=self._run, name="infeng-scheduler", daemon=True)
    self.worker.start()

  def forward(self, session, tokens, *params):
    with self.condition:
      if self.closed: raise RuntimeError("inference engine is closed")
      request = session, tokens, params, Future()
      self.requests.insert(0, request)
      self.condition.notify()
    return request[3].result()

  def _run(self):
    while True:
      with self.condition:
        while not self.requests and not self.closed: self.condition.wait()
        if self.closed:
          for request in self.requests: request[3].set_exception(RuntimeError("inference engine is closed"))
          self.requests.clear()
          return
        params = self.requests[0][2]
        batch = [request for request in self.requests if request[2] == params][:MAX_BATCH]
      try:
        outputs, ready = self.engine.forward_step([request[0] for request in batch], [request[1] for request in batch], *params)
        for request, output, done in zip(batch, outputs, ready):
          if done: request[3].set_result(output)
      except Exception as error:
        for request in batch: request[3].set_exception(error)
        ready = [True] * len(batch)
      with self.condition:
        untouched = [request for request in self.requests if request not in batch]
        self.requests = untouched + [request for request, done in zip(batch, ready) if not done]

  def close(self):
    with self.condition:
      if self.closed: return
      self.closed = True
      self.condition.notify_all()
    self.worker.join()


class Session:

  def __init__(self, engine: InferenceEngine):
    self.engine, self.handle, self.closed = engine, 0, False
    self.handle = _check(_LIB.infeng_sequence_create(engine.handle))
    self.sequence_id, engine._next_sequence = engine._next_sequence, engine._next_sequence + 1
    self.request, self.kv_valid, self.phase, self.outputs = [], 0, "pp", []
    self.speculative, self.draft_count, self.stop_tokens = False, engine.default_draft_tokens, ()
    self.drafted_tokens = self.accepted_tokens = 0
    self.pending, self.sealed = [], True

  def configure(self, *, speculative=False, draft_tokens=None, stop_token_ids=()):
    drafts = self.engine.default_draft_tokens if draft_tokens is None else draft_tokens
    if speculative and self.engine.drafter == "none": raise RuntimeError("speculative decoding requires an active drafter")
    if speculative and not 0 < drafts <= self.engine.max_draft_tokens: raise RuntimeError("draft token count exceeds the active drafter limit")
    self.speculative, self.draft_count, self.stop_tokens = bool(speculative), drafts, tuple(stop_token_ids)

  def _attach(self, ids):
    fresh, generation = not self.request, self.phase == "tg"
    if fresh: self.request = list(ids)
    elif not generation:
      if len(ids) <= self.kv_valid or self.request[:self.kv_valid] != ids[:self.kv_valid]:
        raise RuntimeError("prompt retry must preserve the committed token prefix")
      self.request = list(ids)
    elif len(self.request) == self.kv_valid and self.kv_valid and ids[0] == self.request[-1]: self.request.extend(ids[1:])
    else: self.request = self.request[:self.kv_valid] + list(ids)
    if len(self.request) <= self.kv_valid: raise RuntimeError("request has no uncomputed token")
    if not generation or len(self.request) != self.kv_valid + 1: self.phase = "pp"
    return fresh

  def forward(self, ids, temperature=0.0, top_p=1.0, top_k=None):
    return self.engine._scheduler.forward(self, ids, temperature, top_p, top_k)

  def _tokens(self, message, thinking):
    if self.engine.tokenizer is None: raise RuntimeError("generate requires a tokenizer")
    text = self.engine.tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=thinking)
    prefix = "" if not self.pending else ("\n" if self.sealed else "<|im_end|>\n")
    return list(self.engine.tokenizer.encode(prefix + text, add_special_tokens=False))

  def generate(self, message: str, max_new_tokens: int | None = None, thinking=False, stop_token_ids=None, temperature=0.0, top_p=1.0,
               top_k=None, speculative=False, draft_tokens=None):
    if self.closed: raise RuntimeError("session is closed")
    pending = self.pending + self._tokens(message, thinking)
    self.pending = pending
    remaining = self.engine.max_context - self.kv_valid - len(pending)
    count = remaining if max_new_tokens is None else min(max_new_tokens, remaining)
    if count < 0: raise ValueError(f"context exceeds {self.engine.max_context} tokens")
    im_end = self.engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if stop_token_ids is None: stop_token_ids = [token for token in (self.engine.tokenizer.eos_token_id, im_end) if token is not None]
    drafts = self.engine.default_draft_tokens if draft_tokens is None else draft_tokens
    stopped = False
    try:
      for step in range(count):
        self.configure(speculative=speculative and count - step > 1, draft_tokens=min(drafts, max(1, count - step - 1)), stop_token_ids=stop_token_ids)
        token = self.engine._scheduler.forward(self, pending if step == 0 else pending[0], temperature, top_p, top_k)
        self.pending = pending = [token]
        if token in stop_token_ids:
          self.sealed, stopped = token == im_end, True
          break
        yield token
    finally:
      if not stopped: self.sealed = False

  @property
  def length(self): return self.kv_valid

  @property
  def pending_outputs(self): return len(self.outputs)

  @property
  def mapped_bytes(self): return self.engine.mapped_bytes

  mapped_kv_bytes = mapped_bytes

  def speculative_counters(self):
    return {"drafted_tokens": self.drafted_tokens, "accepted_tokens": self.accepted_tokens,
            "acceptance_rate": self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else None}

  def close(self):
    if not self.closed and self.handle:
      with self.engine._lock: _LIB.infeng_sequence_release(self.handle)
    self.handle, self.closed = 0, True

  def __del__(self):
    self.close()


class InferenceEngine:

  def __init__(self, weights: str | Path, tokenizer: str | None = None, *, drafter="none", draft_weights: str | Path | None = None,
               max_context=65536, profile=False):
    if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
    drafters = {"none": 0, "mtp": 1, "dflash": 2}
    if drafter not in drafters: raise ValueError("drafter must be 'none', 'mtp', or 'dflash'")
    if drafter == "dflash" and draft_weights is None: raise ValueError("dflash requires draft_weights")
    if drafter != "dflash" and draft_weights is not None: raise ValueError("draft_weights is only valid for dflash")
    draft_path = str(draft_weights).encode() if draft_weights is not None else None
    self.handle = _check(_LIB.infeng_model_create(str(weights).encode(), draft_path, str(KERNELS).encode(), max_context, profile, drafters[drafter]))
    self.max_context, self.device, self.dtype, self.drafter = max_context, "metal4", "float16", drafter
    self.has_mtp, self.has_dflash, self.max_batch_sequences, self.vocab_size = drafter == "mtp", drafter == "dflash", MAX_BATCH, VOCAB
    self.max_draft_tokens = 7 if self.has_dflash else 4 if self.has_mtp else 0
    self.default_draft_tokens = 7 if self.has_dflash else 2 if self.has_mtp else 0
    self._lock, self._sessions, self._next_sequence = threading.Lock(), weakref.WeakSet(), 1
    self._scheduler = _Scheduler(self)
    self.tokenizer = None
    if tokenizer:
      os.environ.setdefault("USE_TORCH", "0")
      from transformers import AutoTokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    info = self._info()
    self.parameter_count, self.weight_bytes = info.parameters, info.weight_bytes

  def session(self):
    with self._lock: session = Session(self)
    self._sessions.add(session)
    return session

  def _restore(self, session):
    tokens = (ctypes.c_int32 * len(session.request))(*session.request)
    valid = ctypes.c_uint32()
    _status(_LIB.infeng_prefix(session.handle, tokens, len(session.request), ctypes.byref(valid)))
    session.kv_valid = valid.value

  def _prepare(self, sessions):
    counts, draft_rows, drafts = [1] * len(sessions), 0, self.max_draft_tokens
    for session in sessions:
      if session.phase == "tg" and session.speculative:
        room = min(self.max_context, (session.kv_valid // 512 + 1) * 512) - session.kv_valid
        if room > 1: draft_rows, drafts = draft_rows + 1, min(drafts, session.draft_count, room - 1)
    use_drafts = drafts if draft_rows else 0
    budget, prompts = BLOCK - len(sessions) - draft_rows * use_drafts, sum(session.phase == "pp" for session in sessions)
    left = prompts
    for row, session in enumerate(sessions):
      if session.phase == "pp":
        available = min(len(session.request) - session.kv_valid, (session.kv_valid // 512 + 1) * 512 - session.kv_valid)
        counts[row] += min(available - 1, (budget + left - 1) // left)
        budget -= counts[row] - 1
        left -= 1
    flags = []
    for row, session in enumerate(sessions):
      draft = session.phase == "tg" and session.speculative and use_drafts > 0
      if draft: counts[row] = use_drafts + 1
      sample = draft or session.phase == "tg" or session.kv_valid + counts[row] == len(session.request)
      flags.append(draft | sample << 1)
    return counts, flags, use_drafts

  def forward_step(self, sessions, ids, temperature=0.0, top_p=1.0, top_k=None):
    if not sessions or len(sessions) != len(ids) or len(sessions) > MAX_BATCH: raise ValueError("batch must contain 1 to 8 matching sessions and inputs")
    if len(set(sessions)) != len(sessions) or any(session.engine is not self or session.closed for session in sessions):
      raise ValueError("all sessions must be unique, live, and belong to this model")
    if temperature > 0 and (top_k is not None and not 0 <= top_k <= 64): raise RuntimeError("GPU top_k must be between 1 and 64")
    if temperature > 0 and top_k is None and top_p < 1: raise RuntimeError("top_p below 1 requires top_k on this specialized runtime")
    queries = [[item] if isinstance(item, int) else list(item) for item in ids]
    if any(not query for query in queries): raise ValueError("every query must contain at least one token")
    with self._lock: return self._step(sessions, queries, temperature, top_p, top_k or 0)

  def _step(self, sessions, ids, temperature, top_p, top_k):
    outputs, ready, active, indexes = [0] * len(sessions), [0] * len(sessions), [], []
    for row, (session, query) in enumerate(zip(sessions, ids)):
      if session.outputs:
        if len(query) != 1 or query[0] != session.request[len(session.request) - len(session.outputs) - 1]:
          raise RuntimeError("queued speculative output is out of order")
        outputs[row], ready[row] = session.outputs.pop(0), 1
        continue
      if session._attach(query): self._restore(session)
      if len(session.request) > self.max_context: raise RuntimeError("forward exceeds maximum context")
      active.append(session)
      indexes.append(row)
    if not active: return outputs, ready
    counts, flags, drafts = self._prepare(active)
    segments = [session.request[session.kv_valid:session.kv_valid + (1 if flags[row] & 1 else counts[row])] for row, session in enumerate(active)]
    starts = [0, *accumulate(map(len, segments))]
    handles = (ctypes.c_void_p * len(active))(*(session.handle for session in active))
    tokens = (ctypes.c_int32 * starts[-1])(*(token for segment in segments for token in segment))
    cstarts, valid = (ctypes.c_uint32 * len(starts))(*starts), (ctypes.c_uint32 * len(active))(*(session.kv_valid for session in active))
    ccounts, cflags = (ctypes.c_uint32 * len(active))(*counts), (ctypes.c_uint8 * len(active))(*flags)
    result = (ctypes.POINTER(ctypes.c_int32) * 2)()
    _status(_LIB.infeng_forward(self.handle, handles, tokens, cstarts, valid, ccounts, cflags, len(active), drafts, temperature, top_p, top_k,
                                result))
    sampled, proposed = result
    updates = []
    try:
      sample_row = draft_row = 0
      for row, session in enumerate(active):
        old = session.kv_valid
        if flags[row] & 1:
          candidates = [proposed[(step + 1) * MAX_BATCH + draft_row] for step in range(drafts)]
          take, terminal = 0, False
          for step in range(drafts):
            token = candidates[step]
            if sampled[sample_row + step] != token: break
            take += 1
            if token in session.stop_tokens:
              terminal = True
              break
          request = session.request[:old + 1] + candidates[:take]
          if not terminal: request.append(sampled[sample_row + take])
          current = request[old + 1:]
          updates.append((request, old + 1 + take, take, current))
          sample_row += counts[row]
          draft_row += 1
        else:
          request, current = list(session.request), []
          value = old + counts[row]
          if flags[row] & 2:
            request.append(sampled[sample_row])
            current = [request[-1]]
            sample_row += 1
          updates.append((request, value, 0, current))
      offsets = [session.kv_valid // BLOCK * BLOCK for session in active]
      segments = [request[offset:value] for offset, (request, value, _, _) in zip(offsets, updates)]
      commit_starts = [0, *accumulate(map(len, segments))]
      commit_tokens = (ctypes.c_int32 * commit_starts[-1])(*(token for segment in segments for token in segment))
      _status(_LIB.infeng_commit(self.handle, commit_tokens, (ctypes.c_uint32 * len(commit_starts))(*commit_starts),
                                 (ctypes.c_uint32 * len(offsets))(*offsets),
                                 (ctypes.c_uint32 * len(updates))(*(update[1] for update in updates)),
                                 (ctypes.c_uint8 * len(updates))(*(update[2] for update in updates))))
    except Exception:
      _LIB.infeng_abort(self.handle)
      raise
    for row, session in enumerate(active):
      request, valid, accepted, current = updates[row]
      session.request, session.kv_valid, session.phase = request, valid, "tg" if flags[row] & 2 else session.phase
      if flags[row] & 1:
        session.drafted_tokens += drafts
        session.accepted_tokens += accepted
      if current:
        output = current.pop(0)
        outputs[indexes[row]], ready[indexes[row]], session.outputs = output, 1, current
    return outputs, ready

  def forward_batch(self, sessions, ids, temperature=0.0, top_p=1.0, top_k=None):
    pending, outputs = list(range(len(sessions))), [None] * len(sessions)
    while pending:
      values, ready = self.forward_step([sessions[i] for i in pending], [ids[i] for i in pending], temperature, top_p, top_k)
      remaining = []
      for row, index in enumerate(pending):
        if ready[row]: outputs[index] = values[row]
        else: remaining.append(index)
      pending = remaining
    return outputs

  def _info(self):
    value = _Info()
    _LIB.infeng_info(self.handle, ctypes.byref(value))
    return value

  @property
  def mapped_bytes(self): return self._info().mapped_bytes

  def counters(self):
    info = self._info()
    return {"gpu_time_ns": info.gpu_time_ns, "passes": info.passes}

  def kernel_counters(self):
    result = []
    for index in range(_LIB.infeng_kernel_counter_count(self.handle)):
      phase, name, gpu, launches = ctypes.c_char_p(), ctypes.c_char_p(), ctypes.c_uint64(), ctypes.c_uint64()
      _LIB.infeng_kernel_counter(self.handle, index, ctypes.byref(phase), ctypes.byref(name), ctypes.byref(gpu), ctypes.byref(launches))
      result.append({"phase": phase.value.decode(), "name": name.value.decode(), "gpu_time_ns": gpu.value, "launches": launches.value})
    return result

  def close(self):
    if getattr(self, "_scheduler", None):
      self._scheduler.close()
      self._scheduler = None
    for session in list(getattr(self, "_sessions", ())): session.close()
    if getattr(self, "handle", 0):
      _LIB.infeng_model_release(self.handle)
      self.handle = 0

  def __del__(self):
    self.close()
