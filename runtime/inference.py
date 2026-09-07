import ctypes
import os
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LIB, KERNELS = ROOT / ".build/libinfeng.dylib", ROOT / "backend/metal/kernel"
SOURCES = [ROOT / path for path in ("backend/metal/device.cpp", "model/qwen35/model.cpp", "model/qwen35/forward.cpp", "runtime/engine.cpp")]
MAX_BATCH, MAX_DRAFTS, VOCAB = 8, 7, 248320
class _Info(ctypes.Structure):
  _fields_ = ([(name, ctypes.c_uint64) for name in ("parameters", "weight_bytes", "mapped_bytes", "drafted", "accepted")] +
              [("valid", ctypes.c_uint32)])
def _load():
  headers = [*ROOT.glob("backend/metal/*.hpp"), *(ROOT / "model").rglob("*.hpp"), *(ROOT / "third_party/metal-cpp").rglob("*.hpp")]
  if not LIB.exists() or any(path.stat().st_mtime > LIB.stat().st_mtime for path in [*SOURCES, *headers]):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    ggml = subprocess.check_output(["pkg-config", "--cflags", "--libs", "ggml"], text=True).split()
    subprocess.run(["xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-fblocks", "-fvisibility=hidden", "-DMETALCPP_SYMBOL_VISIBILITY_HIDDEN",
                    "-dynamiclib", *map(str, SOURCES), "-I", str(ROOT / "third_party/metal-cpp"), "-I", str(ROOT), *ggml, "-lggml-base",
                    "-framework", "Foundation", "-framework", "Metal", "-o", str(LIB)], check=True)
  lib = ctypes.CDLL(LIB)
  void = ctypes.c_void_p
  lib.infeng_engine_create.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p)
  lib.infeng_engine_create.restype = void
  lib.infeng_engine_release.argtypes = (void, )
  lib.infeng_sequence_create.argtypes = (void, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_float, ctypes.c_float,
                                         ctypes.c_int32, ctypes.c_uint32, ctypes.c_char_p)
  lib.infeng_sequence_create.restype = void
  lib.infeng_sequence_release.argtypes = (void, )
  lib.infeng_sequence_append.argtypes, lib.infeng_sequence_append.restype = (void, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32), ctypes.c_int32
  lib.infeng_sequence_read.argtypes, lib.infeng_sequence_read.restype = (void, ctypes.c_uint32), ctypes.c_int32
  lib.infeng_engine_info.argtypes, lib.infeng_engine_info.restype = (void, ctypes.c_uint32), ctypes.c_uint64
  lib.infeng_sequence_info.argtypes, lib.infeng_sequence_info.restype = (void, ctypes.c_uint32), ctypes.c_uint64
  return lib
_LIB = _load()
def _check(value, error=None):
  errors = {-3: "sequence is still generating", -4: "completion has no available model context", -5: "KV block pool has no evictable capacity"}
  if value is None or value == -2: raise RuntimeError((error.value or b"native Metal operation failed").decode())
  if value in errors: raise RuntimeError(errors[value])
  return value
def _snapshot(engine, sequence=None):
  values = [_LIB.infeng_engine_info(engine, field) for field in range(3)]
  values += [_LIB.infeng_sequence_info(sequence, field) for field in range(3)] if sequence else [0] * 3
  return _Info(*values)
class Sequence:
  def __init__(self, engine, stop_token_ids, temperature, top_p, top_k, draft_tokens):
    self.engine, self.handle = engine, 0
    self.stop_token_ids, self.draft_tokens = frozenset(stop_token_ids), draft_tokens
    stops = (ctypes.c_int32 * len(stop_token_ids))(*stop_token_ids)
    error = ctypes.create_string_buffer(1024)
    self.handle = _check(_LIB.infeng_sequence_create(engine.handle, stops, len(stops), temperature, top_p, top_k or 0, draft_tokens, error), error)
  def append(self, prompt=""):
    if isinstance(prompt, str):
      if prompt and self.engine.tokenizer is None: raise RuntimeError("string completion requires a tokenizer")
      prompt = self.engine.tokenizer.encode(prompt, add_special_tokens=False) if prompt else ()
    tokens = list(prompt)
    ids = (ctypes.c_int32 * len(tokens))(*tokens)
    return _check(_LIB.infeng_sequence_append(self.handle, ids, len(tokens)))
  def read(self, cursor):
    value = _check(_LIB.infeng_sequence_read(self.handle, cursor))
    return None if value == -1 else value
  def cancel(self):
    if self.handle: _LIB.infeng_sequence_append(self.handle, None, 0xffffffff)
  @property
  def length(self): return _snapshot(self.engine.handle, self.handle).valid
  def speculative_counters(self):
    info = _snapshot(self.engine.handle, self.handle)
    return {"drafted_tokens": info.drafted, "accepted_tokens": info.accepted,
            "acceptance_rate": info.accepted / info.drafted if info.drafted else None}
  def close(self):
    if self.handle: _LIB.infeng_sequence_release(self.handle)
    self.handle = 0
  def __del__(self): self.close()
class InferenceEngine:
  def __init__(self, weights: str | Path, tokenizer: str | None = None, *, drafter="none", draft_weights: str | Path | None = None,
               max_context=65536):
    drafters = {"none": (0, 0, 0), "mtp": (1, 4, 2), "dflash": (2, MAX_DRAFTS, MAX_DRAFTS)}
    if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
    if drafter not in drafters: raise ValueError("drafter must be 'none', 'mtp', or 'dflash'")
    if drafter == "dflash" and draft_weights is None: raise ValueError("dflash requires draft_weights")
    if drafter != "dflash" and draft_weights is not None: raise ValueError("draft_weights is only valid for dflash")
    native, max_drafts, default_drafts = drafters[drafter]
    draft_path = str(draft_weights).encode() if draft_weights is not None else None
    error = ctypes.create_string_buffer(1024)
    self.handle = _check(_LIB.infeng_engine_create(str(weights).encode(), draft_path, str(KERNELS).encode(), max_context, native, error), error)
    self.max_context, self.drafter, self.max_batch_sequences, self.vocab_size = max_context, drafter, MAX_BATCH, VOCAB
    self.max_draft_tokens, self.default_draft_tokens = max_drafts, default_drafts
    self.tokenizer = None
    if tokenizer:
      os.environ.setdefault("USE_TORCH", "0")
      from transformers import AutoTokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    info = _snapshot(self.handle)
    self.parameter_count, self.weight_bytes = info.parameters, info.weight_bytes
  def sequence(self, *, stop_token_ids=None, temperature=0.0, top_p=1.0, top_k=None, draft_tokens=0):
    if not 0 <= draft_tokens <= self.max_draft_tokens: raise RuntimeError("draft token count exceeds the active drafter limit")
    if temperature > 0 and not 0 <= (top_k or 0) <= 64: raise RuntimeError("GPU top_k must be between 0 and 64")
    if temperature > 0 and top_p < 1 and not top_k: raise RuntimeError("top_p below 1 requires top_k on this specialized runtime")
    if stop_token_ids is None and self.tokenizer:
      im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
      stop_token_ids = [token for token in (self.tokenizer.eos_token_id, im_end) if token is not None]
    sequence = Sequence(self, list(stop_token_ids or ()), temperature, top_p, top_k, draft_tokens)
    return sequence
  @property
  def mapped_bytes(self): return _snapshot(self.handle).mapped_bytes
  def close(self):
    if getattr(self, "handle", 0): _LIB.infeng_engine_release(self.handle)
    self.handle = 0
  def __del__(self): self.close()
