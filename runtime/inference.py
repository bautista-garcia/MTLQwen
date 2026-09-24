import ctypes
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB, KERNELS = ROOT / ".build/libinfeng.dylib", ROOT / "backend/metal/kernel"
SOURCES = [ROOT / path for path in ("backend/metal/device.cpp", "model/qwen35/forward.cpp", "runtime/engine.cpp")]
MAX_BATCH, VOCAB = 8, 248320


def _load():
  # rebuilds c++ engine (only if src has changed)
  headers = [*(ROOT / "backend/metal").rglob("*.hpp"), *(ROOT / "model").rglob("*.hpp"), *(ROOT / "third_party/metal-cpp").rglob("*.hpp")]
  if not LIB.exists() or any(path.stat().st_mtime > LIB.stat().st_mtime for path in [*SOURCES, *headers]):
    LIB.parent.mkdir(parents=True, exist_ok=True)
    ggml = subprocess.check_output(["pkg-config", "--cflags", "--libs", "ggml"], text=True).split()
    subprocess.run([
      "xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-fblocks", "-fvisibility=hidden", "-DMETALCPP_SYMBOL_VISIBILITY_HIDDEN", "-dynamiclib",
      *map(str, SOURCES), "-I",
      str(ROOT / "third_party/metal-cpp"), "-I",
      str(ROOT), *ggml, "-lggml-base", "-framework", "Foundation", "-framework", "Metal", "-o",
      str(LIB)
    ],
                   check=True)
  # define python-c++ argument interface
  lib = ctypes.CDLL(LIB)
  void = ctypes.c_void_p
  lib.infeng_engine_create.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p)
  lib.infeng_engine_create.restype = void
  lib.infeng_engine_release.argtypes, lib.infeng_engine_release.restype = (void, ), None
  lib.infeng_sequence_create.argtypes = (void, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                                         ctypes.c_uint32, ctypes.c_char_p)
  lib.infeng_sequence_create.restype = void
  lib.infeng_sequence_release.argtypes, lib.infeng_sequence_release.restype = (void, ), None
  lib.infeng_sequence_append.argtypes, lib.infeng_sequence_append.restype = (void, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32), ctypes.c_int32
  lib.infeng_sequence_read.argtypes, lib.infeng_sequence_read.restype = (void, ctypes.c_uint32), ctypes.c_int32
  lib.infeng_engine_info.argtypes, lib.infeng_engine_info.restype = (void, ctypes.c_uint32), ctypes.c_uint64
  lib.infeng_sequence_info.argtypes, lib.infeng_sequence_info.restype = (void, ctypes.c_uint32), ctypes.c_uint64
  return lib


_LIB = _load()


# c++ error code translation
def _check(value, error=None):
  errors = {-3: "sequence is still generating", -4: "completion has no available model context", -5: "KV block pool has no evictable capacity"}
  if value is None: raise RuntimeError((error.value or b"native Metal operation failed").decode())
  if value in errors: raise RuntimeError(errors[value])
  return value


class Sequence:

  def __init__(self, engine, stop_token_ids, temperature, top_p, top_k, speculative):
    self.engine, self.handle = engine, 0
    self.stop_token_ids, self.speculative = frozenset(stop_token_ids), bool(speculative and engine.draft_width)
    stops = (ctypes.c_int32 * len(stop_token_ids))(*stop_token_ids)
    error = ctypes.create_string_buffer(1024)
    self.handle = _check(_LIB.infeng_sequence_create(engine.handle, stops, len(stops), temperature, top_p, top_k or 0, self.speculative, error),
                         error)

  def append(self, tokens=()):
    ids = (ctypes.c_int32 * len(tokens))(*tokens)
    return _check(_LIB.infeng_sequence_append(self.handle, ids, len(tokens)))

  def read(self, cursor):
    value = _check(_LIB.infeng_sequence_read(self.handle, cursor))
    return None if value == -1 else value

  def cancel(self):
    if self.handle: _LIB.infeng_sequence_append(self.handle, None, 0xffffffff)

  @property
  def length(self):
    return _LIB.infeng_sequence_info(self.handle, 2)

  def speculative_counters(self):
    drafted = _LIB.infeng_sequence_info(self.handle, 0)
    accepted = _LIB.infeng_sequence_info(self.handle, 1)
    return {"drafted_tokens": drafted, "accepted_tokens": accepted, "acceptance_rate": accepted / drafted if drafted else None}

  def close(self):
    if self.handle: _LIB.infeng_sequence_release(self.handle)
    self.handle = 0

  def __del__(self):
    self.close()


class InferenceEngine:

  def __init__(self, weights: str | Path, *, draft_weights: str | Path | None = None, max_context=65536):
    if not 0 < max_context <= 65536: raise ValueError(f"max_context must be between 1 and 65536, got {max_context}")
    draft_path = str(draft_weights).encode() if draft_weights is not None else None
    error = ctypes.create_string_buffer(1024)
    self.handle = _check(_LIB.infeng_engine_create(str(weights).encode(), draft_path, str(KERNELS).encode(), max_context, error), error)
    self.drafter = ("none", "mtp", "dflash")[_LIB.infeng_engine_info(self.handle, 3)]
    self.draft_width = _LIB.infeng_engine_info(self.handle, 4)
    self.max_context, self.max_batch_sequences, self.vocab_size = max_context, MAX_BATCH, VOCAB
    self.parameter_count = _LIB.infeng_engine_info(self.handle, 0)
    self.weight_bytes = _LIB.infeng_engine_info(self.handle, 1)

  def sequence(self, *, stop_token_ids=None, temperature=0.0, top_p=1.0, top_k=None, speculative=False):
    return Sequence(self, list(stop_token_ids or ()), temperature, top_p, top_k, speculative)

  @property
  def mapped_bytes(self):
    return _LIB.infeng_engine_info(self.handle, 2)

  def close(self):
    if getattr(self, "handle", 0): _LIB.infeng_engine_release(self.handle)
    self.handle = 0

  def __del__(self):
    self.close()
