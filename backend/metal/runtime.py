from __future__ import annotations

import ctypes
import subprocess
import weakref
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / ".build/libinfeng.dylib"
KERNELS = ROOT / "backend/metal/kernel"
SOURCES = [ROOT / "backend/metal/device.cpp", ROOT / "model/qwen35_weights.cpp", ROOT / "model/qwen35_state.cpp",
           ROOT / "model/qwen35_ops.cpp", ROOT / "model/qwen35.cpp", ROOT / "model/qwen35_api.cpp"]


def _build():
    headers = [*ROOT.glob("backend/metal/*.hpp"), *ROOT.glob("model/*.hpp"),
               *(ROOT / "third_party/metal-cpp").rglob("*.hpp")]
    if LIB.exists() and all(path.stat().st_mtime <= LIB.stat().st_mtime for path in [*SOURCES, *headers]): return
    LIB.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["xcrun", "clang++", "-std=c++17", "-O3", "-DNDEBUG", "-fblocks", "-fvisibility=hidden",
                    "-DMETALCPP_SYMBOL_VISIBILITY_HIDDEN", "-dynamiclib",
                    *(str(path) for path in SOURCES), "-I", str(ROOT / "third_party/metal-cpp"), "-I", str(ROOT),
                    "-framework", "Foundation", "-framework", "Metal", "-o", str(LIB)], check=True)


class _Counters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("gpu_time_ns", "passes")]


class _SpecCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ("drafted_tokens", "accepted_tokens")]


def _load():
    _build(); lib = ctypes.CDLL(LIB)
    lib.infeng_last_error.restype = ctypes.c_char_p
    lib.infeng_model_create.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int32)
    lib.infeng_model_create.restype = ctypes.c_void_p
    lib.infeng_model_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_session_create.argtypes = (ctypes.c_void_p,); lib.infeng_session_create.restype = ctypes.c_void_p
    lib.infeng_session_release.argtypes = (ctypes.c_void_p,)
    lib.infeng_session_configure.argtypes = (ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32,
                                             ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32)
    lib.infeng_forward_batch.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
                                         ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_float,
                                         ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                                         ctypes.POINTER(ctypes.c_uint8))
    lib.infeng_session_length.argtypes = (ctypes.c_void_p,); lib.infeng_session_length.restype = ctypes.c_uint64
    lib.infeng_session_id.argtypes = (ctypes.c_void_p,); lib.infeng_session_id.restype = ctypes.c_uint64
    lib.infeng_session_pending_outputs.argtypes = (ctypes.c_void_p,)
    lib.infeng_session_pending_outputs.restype = ctypes.c_uint32
    lib.infeng_session_mapped_bytes.argtypes = (ctypes.c_void_p,); lib.infeng_session_mapped_bytes.restype = ctypes.c_uint64
    for name in ("infeng_model_parameter_count", "infeng_model_weight_bytes", "infeng_model_vocab_size"):
        function = getattr(lib, name); function.argtypes = (ctypes.c_void_p,); function.restype = ctypes.c_uint64
    lib.infeng_model_counters.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Counters))
    lib.infeng_model_has_mtp.argtypes = (ctypes.c_void_p,); lib.infeng_model_has_mtp.restype = ctypes.c_int32
    lib.infeng_session_spec_counters.argtypes = (ctypes.c_void_p, ctypes.POINTER(_SpecCounters))
    lib.infeng_model_kernel_counter_count.argtypes = (ctypes.c_void_p,); lib.infeng_model_kernel_counter_count.restype = ctypes.c_uint32
    lib.infeng_model_kernel_counter.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
                                                ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64),
                                                ctypes.POINTER(ctypes.c_uint64))
    return lib


_LIB = _load()


def _error(): return (_LIB.infeng_last_error() or b"native Metal operation failed").decode()
def _check(value):
    if not value: raise RuntimeError(_error())
    return value
def _status(value):
    if value: raise RuntimeError(_error())


class NativeSession:
    def __init__(self, model: NativeModel):
        self.model, self.handle, self.closed = model, 0, False
        self.handle = _check(_LIB.infeng_session_create(model.handle))
        self._configuration = None

    def configure(self, *, speculative=False, draft_tokens=2, stop_token_ids=()):
        configuration = bool(speculative), draft_tokens, tuple(stop_token_ids)
        if configuration == self._configuration: return
        stops = array("i", configuration[2]); pointer = None
        if stops: pointer = (ctypes.c_int32 * len(stops)).from_buffer(stops)
        _status(_LIB.infeng_session_configure(self.handle, speculative, draft_tokens, pointer, len(stops)))
        self._configuration = configuration

    def forward(self, ids, temperature=0.0, top_p=1.0, top_k=None):
        while True:
            outputs, ready = self.model.forward_step([self], [ids], temperature, top_p, top_k)
            if ready[0]: return outputs[0]

    @property
    def length(self): return _LIB.infeng_session_length(self.handle)

    @property
    def sequence_id(self): return _LIB.infeng_session_id(self.handle)

    @property
    def pending_outputs(self): return _LIB.infeng_session_pending_outputs(self.handle)

    @property
    def mapped_bytes(self): return _LIB.infeng_session_mapped_bytes(self.handle)

    def speculative_counters(self):
        output = _SpecCounters(); _status(_LIB.infeng_session_spec_counters(self.handle, ctypes.byref(output)))
        result = {name: getattr(output, name) for name, _ in output._fields_}
        result["acceptance_rate"] = (result["accepted_tokens"] / result["drafted_tokens"]
                                     if result["drafted_tokens"] else None)
        return result

    def close(self):
        if not self.closed and self.handle: _LIB.infeng_session_release(self.handle)
        self.handle, self.closed = 0, True

    def __del__(self): self.close()


class NativeModel:
    def __init__(self, weights: str | Path, *, max_context=65536, profile=False):
        self._sessions = weakref.WeakSet()
        self.handle = 0
        self.handle = _check(_LIB.infeng_model_create(str(weights).encode(), str(KERNELS).encode(), max_context, profile))

    def session(self):
        session = NativeSession(self); self._sessions.add(session); return session

    def forward_step(self, sessions, ids, temperature=0.0, top_p=1.0, top_k=None):
        if len(sessions) != len(ids): raise ValueError("sessions and ids must have equal length")
        if not sessions: raise ValueError("batch is empty")
        if any(session.model is not self or session.closed for session in sessions):
            raise ValueError("all sessions must be live and belong to this model")
        handles = (ctypes.c_void_p * len(sessions))(*(session.handle for session in sessions))
        queries = [[item] if isinstance(item, int) else list(item) for item in ids]
        if any(not query for query in queries): raise ValueError("every query must contain at least one token")
        starts = [0]
        for query in queries: starts.append(starts[-1] + len(query))
        inputs = (ctypes.c_int32 * starts[-1])(*(token for query in queries for token in query))
        offsets = (ctypes.c_uint32 * len(starts))(*starts)
        outputs, ready = (ctypes.c_int32 * len(ids))(), (ctypes.c_uint8 * len(ids))()
        _status(_LIB.infeng_forward_batch(handles, inputs, offsets, len(ids), temperature, top_p, top_k or 0,
                                          outputs, ready))
        return list(outputs), list(ready)

    def forward_batch(self, sessions, ids, temperature=0.0, top_p=1.0, top_k=None):
        if len(sessions) != len(ids): raise ValueError("sessions and ids must have equal length")
        if not sessions: raise ValueError("batch is empty")
        pending, outputs = list(range(len(sessions))), [None] * len(sessions)
        while pending:
            tokens, done = self.forward_step([sessions[i] for i in pending], [ids[i] for i in pending],
                                             temperature, top_p, top_k)
            remaining = []
            for row, index in enumerate(pending):
                if done[row]: outputs[index] = tokens[row]
                else: remaining.append(index)
            pending = remaining
        return outputs

    @property
    def parameter_count(self): return _LIB.infeng_model_parameter_count(self.handle)

    @property
    def weight_bytes(self): return _LIB.infeng_model_weight_bytes(self.handle)

    @property
    def vocab_size(self): return _LIB.infeng_model_vocab_size(self.handle)

    @property
    def has_mtp(self): return bool(_LIB.infeng_model_has_mtp(self.handle))

    def counters(self):
        output = _Counters(); _status(_LIB.infeng_model_counters(self.handle, ctypes.byref(output)))
        return {name: getattr(output, name) for name, _ in output._fields_}

    def kernel_counters(self):
        counters = []
        for index in range(_LIB.infeng_model_kernel_counter_count(self.handle)):
            phase = ctypes.c_char_p(); name = ctypes.c_char_p(); gpu_time_ns = ctypes.c_uint64(); launches = ctypes.c_uint64()
            _status(_LIB.infeng_model_kernel_counter(self.handle, index, ctypes.byref(phase), ctypes.byref(name),
                                                     ctypes.byref(gpu_time_ns), ctypes.byref(launches)))
            counters.append({"phase": phase.value.decode(), "name": name.value.decode(),
                             "gpu_time_ns": gpu_time_ns.value, "launches": launches.value})
        return counters

    def close(self):
        for session in list(self._sessions): session.close()
        if getattr(self, "handle", 0): _LIB.infeng_model_release(self.handle); self.handle = 0

    def __del__(self): self.close()
