from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import InferenceEngine  # noqa: E402


class ChatSession:
  """CPU-owned conversation metadata paired lazily with one native sequence."""

  def __init__(self):
    self.id, self.title = uuid.uuid4().hex[:12], "New chat"
    self.runtime, self.messages, self.metrics = None, [], {}
    self.updated_at = time.time()
    self.generating, self.started, self.sealed, self.cancel = False, False, True, threading.Event()
    self.lock = threading.Lock()

  def summary(self):
    runtime = self.runtime
    return {
      "id": self.id,
      "title": self.title,
      "updated_at": self.updated_at,
      "generating": self.generating,
      "message_count": len(self.messages),
      "metrics": self.metrics,
      "sequence_id": self.id if runtime else None,
      "mapped_kv_bytes": runtime.engine.mapped_bytes if runtime else 0
    }

class Handler(SimpleHTTPRequestHandler):
  protocol_version = "HTTP/1.1"
  disable_nagle_algorithm, wbufsize = True, 0
  engine = None
  loading, load_error = False, ""
  load_config = None
  model_condition = threading.Condition()
  sessions = {}
  max_sessions = 8

  @classmethod
  def _model_config(cls, drafter=None):
    drafter = drafter or cls.args.drafter
    if drafter not in ("none", "mtp", "dflash"): raise ValueError("drafter must be none, mtp, or dflash")
    weights = cls.args.mtp_weights if drafter == "mtp" else cls.args.weights
    return {"drafter": drafter, "weights": weights, "draft_weights": cls.args.draft_weights if drafter == "dflash" else None}

  @classmethod
  def _finish_model_load(cls, config):
    try:
      print(f"loading {config['drafter']} model on native Metal 4 with float16", flush=True)
      engine = InferenceEngine(config["weights"],
                               cls.args.tokenizer,
                               drafter=config["drafter"],
                               draft_weights=config["draft_weights"],
                               max_context=cls.args.max_context)
      error = ""
      print("model loaded on native Metal 4 with float16", flush=True)
    except Exception as exc:
      engine, error = None, f"{type(exc).__name__}: {exc}"
      print(f"model load failed: {error}", flush=True)
    with cls.model_condition:
      cls.engine, cls.loading, cls.load_error = engine, False, error
      cls.model_condition.notify_all()
    return engine

  @classmethod
  def load_model(cls):
    with cls.model_condition:
      if cls.engine: return cls.engine
      if cls.loading:
        while cls.loading:
          cls.model_condition.wait()
        if cls.engine: return cls.engine
        raise RuntimeError(cls.load_error or "model load failed")
      cls.loading, cls.load_error = True, ""
      config = cls.load_config or cls._model_config()
      cls.load_config = config
    engine = cls._finish_model_load(config)
    if not engine: raise RuntimeError(cls.load_error)
    return engine

  @classmethod
  def start_model_load(cls, drafter=None):
    config = cls._model_config(drafter)
    with cls.model_condition:
      if cls.engine or cls.loading: return False
      cls.loading, cls.load_error = True, ""
      cls.load_config = config
    threading.Thread(target=cls._finish_model_load, args=(config, ), name="infeng-loader", daemon=True).start()

  def _json(self, status, body):
    payload = json.dumps(body).encode()
    self.send_response(status)
    self.send_header("content-type", "application/json")
    self.send_header("content-length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

  def _body(self):
    size = int(self.headers.get("content-length", 0))
    return json.loads(self.rfile.read(size)) if size else {}

  def _event(self, body):
    try:
      self.wfile.write(f"data: {json.dumps(body, ensure_ascii=False, separators=(',', ':'))}\n\n".encode())
      self.wfile.flush()
      return True
    except (BrokenPipeError, ConnectionResetError):
      return False

  def do_GET(self):
    path = urlparse(self.path).path
    if path == "/": self.path, path = "/index.html", "/index.html"
    if path == "/api/status":
      sessions = list(Handler.sessions.values())
      mapped = max((item.runtime.engine.mapped_bytes for item in sessions if item.runtime), default=0)
      active = sum(item.generating for item in sessions)
      drafter = Handler.engine.drafter if Handler.engine else (Handler.load_config or Handler._model_config())["drafter"]
      default_drafts = Handler.engine.default_draft_tokens if Handler.engine else {"none": 0, "mtp": 2, "dflash": 7}[drafter]
      max_drafts = Handler.engine.max_draft_tokens if Handler.engine else {"none": 0, "mtp": 4, "dflash": 7}[drafter]
      return self._json(
        200, {
          "loaded": Handler.engine is not None,
          "loading": Handler.loading,
          "error": Handler.load_error,
          "device": "metal4" if Handler.engine else "",
          "drafter": drafter,
          "mtp": drafter == "mtp",
          "dflash": drafter == "dflash",
          "default_draft_tokens": default_drafts,
          "max_draft_tokens": max_drafts,
          "sessions": len(sessions),
          "active_sessions": active,
          "max_sessions": Handler.max_sessions,
          "mapped_kv_bytes": mapped
        })
    if path == "/api/sessions":
      body = sorted((item.summary() for item in Handler.sessions.values()), key=lambda item: item["updated_at"], reverse=True)
      return self._json(200, body)
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[:2] == ["api", "sessions"]:
      session = Handler.sessions.get(parts[2])
      return self._json(200, {**session.summary(), "messages": session.messages}) if session else self._json(404, {"error": "session not found"})
    return super().do_GET()

  def do_POST(self):
    path = urlparse(self.path).path
    if path == "/api/load":
      data = self._body()
      drafter = data.get("drafter", Handler.args.drafter)
      Handler.start_model_load(drafter)
      return self._json(200, {"ok": True, "loaded": Handler.engine is not None, "loading": Handler.loading})
    if path == "/api/sessions":
      if len(Handler.sessions) >= Handler.max_sessions:
        return self._json(409, {"error": f"at most {Handler.max_sessions} live sessions are supported"})
      session = ChatSession()
      Handler.sessions[session.id] = session
      return self._json(201, {**session.summary(), "messages": session.messages})
    parts = path.strip("/").split("/")
    if len(parts) != 4 or parts[:2] != ["api", "sessions"]:
      return self._json(404, {"error": "not found"})
    session = Handler.sessions.get(parts[2])
    if not session: return self._json(404, {"error": "session not found"})
    if parts[3] == "cancel":
      session.cancel.set()
      if session.runtime: session.runtime.cancel()
      return self._json(200, {"ok": True})
    if parts[3] == "chat": return self._chat(session, self._body())
    return self._json(404, {"error": "not found"})

  def do_DELETE(self):
    parts = urlparse(self.path).path.strip("/").split("/")
    if len(parts) != 3 or parts[:2] != ["api", "sessions"]:
      return self._json(404, {"error": "not found"})
    session = Handler.sessions.get(parts[2])
    if not session: return self._json(404, {"error": "session not found"})
    if session.generating: return self._json(409, {"error": "stop generation before deleting this session"})
    del Handler.sessions[parts[2]]
    if session.runtime: session.runtime.close()
    return self._json(200, {"ok": True})

  def _chat(self, session, data):
    message = data.get("message", "")
    if not isinstance(message, str) or not message.strip(): return self._json(400, {"error": "message is required"})
    if not session.lock.acquire(blocking=False): return self._json(409, {"error": "this session is already generating"})
    session.generating, disconnected = True, False
    session.cancel.clear()
    started = time.perf_counter()
    try:
      self.send_response(200)
      self.send_header("content-type", "text/event-stream; charset=utf-8")
      self.send_header("x-accel-buffering", "no")
      self.send_header("connection", "close")
      self.end_headers()
      if Handler.engine is None and not self._event({"status": "Loading model…"}): return
      engine = Handler.load_model()
      text = message.strip()
      session.messages.append({"role": "user", "content": text})
      if session.title == "New chat": session.title = text[:48] + ("…" if len(text) > 48 else "")
      session.updated_at = time.time()
      thinking = bool(data.get("thinking", Handler.args.thinking))
      max_tokens = max(1, min(int(data.get("max_tokens", 1024)), engine.max_context))
      if session.runtime is None:
        speculative = bool(data.get("speculative", Handler.args.speculative))
        if speculative and engine.drafter == "none": raise ValueError("speculative generation requires --drafter mtp or dflash")
        requested = data.get("draft_tokens", Handler.args.draft_tokens)
        drafts = engine.default_draft_tokens if requested is None else int(requested)
        if engine.max_draft_tokens: drafts = max(1, min(drafts, engine.max_draft_tokens))
        session.runtime = engine.sequence(temperature=float(data.get("temperature", Handler.args.temperature)),
                                          top_p=float(data.get("top_p", Handler.args.top_p)),
                                          top_k=int(data.get("top_k", Handler.args.top_k)),
                                          draft_tokens=drafts if speculative else 0)
      speculative, stop_ids = bool(session.runtime.draft_tokens), session.runtime.stop_token_ids
      before_context = session.metrics.get("context_tokens", 0)
      spec_before = session.runtime.speculative_counters() if speculative else None
      formatted = engine.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                                       enable_thinking=thinking)
      formatted = ("" if not session.started else "\n" if session.sealed else "<|im_end|>\n") + formatted
      session.started = True
      cursor = session.runtime.append(formatted)
      prompt_tokens = cursor - before_context
      if not self._event({"status": "Generating…", "sequence_id": session.id}):
        disconnected = True
        session.cancel.set()
        session.runtime.cancel()

      # Stream token deltas; the authoritative response is joined once when generation finishes.
      thought, response, mode = [], [], "thinking" if thinking else "response"
      first_at, generated, stop_token, stopping = None, 0, None, False
      while True:
        token_id = session.runtime.read(cursor)
        if token_id is None: break
        cursor += 1
        if token_id in stop_ids:
          stop_token = token_id
          break
        now = time.perf_counter()
        if first_at is None: first_at = now
        generated += 1
        token = engine.tokenizer.decode([token_id], skip_special_tokens=False)
        if mode == "thinking" and "</think>" in token:
          before, after = token.split("</think>", 1)
          thought.append(before)
          response.append(after)
          delta, mode = {"thinking_delta": before, "response_delta": after}, "response"
        elif mode == "thinking":
          thought.append(token)
          delta = {"thinking_delta": token}
        else:
          response.append(token)
          delta = {"response_delta": token}
        decode_elapsed = now - first_at
        metrics = {
          "prompt_tokens": prompt_tokens,
          "generated_tokens": generated,
          "context_tokens": cursor,
          "ttft_ms": round((first_at - started) * 1000, 1),
          "elapsed_ms": round((now - started) * 1000, 1),
          "tps": round((generated - 1) / decode_elapsed, 2) if generated > 1 and decode_elapsed else None,
          "speculative": speculative
        }
        session.metrics = metrics
        if not disconnected and not self._event({**delta, "metrics": metrics}):
          disconnected = True
          session.cancel.set()
          session.runtime.cancel()
        if not stopping and (generated >= max_tokens or session.cancel.is_set()):
          stopping = True
          session.runtime.cancel()

      now = time.perf_counter()
      decode_elapsed = now - first_at if first_at else 0
      spec = {"speculative": speculative, "drafted_tokens": 0, "accepted_tokens": 0, "acceptance_rate": None}
      if spec_before:
        current = session.runtime.speculative_counters()
        spec["drafted_tokens"] = current["drafted_tokens"] - spec_before["drafted_tokens"]
        spec["accepted_tokens"] = current["accepted_tokens"] - spec_before["accepted_tokens"]
        spec["acceptance_rate"] = (round(spec["accepted_tokens"] / spec["drafted_tokens"], 3) if spec["drafted_tokens"] else None)
      metrics = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "context_tokens": cursor,
        "ttft_ms": round(((first_at or now) - started) * 1000, 1),
        "elapsed_ms": round((now - started) * 1000, 1),
        "tps": round((generated - 1) / decode_elapsed, 2) if generated > 1 and decode_elapsed else None,
        **spec
      }
      cancelled = stopping or session.cancel.is_set()
      session.sealed = stop_token == engine.tokenizer.convert_tokens_to_ids("<|im_end|>")
      thought, response = "".join(thought), "".join(response)
      session.metrics = metrics
      session.messages.append({"role": "assistant", "content": response, "thinking": thought, "cancelled": cancelled, "metrics": metrics})
      session.updated_at = time.time()
      if not disconnected: self._event({"done": True, "cancelled": cancelled, "metrics": metrics})
    except Exception as exc:
      if not disconnected: self._event({"error": f"{type(exc).__name__}: {exc}"})
    finally:
      session.generating = False
      session.cancel.clear()
      session.updated_at = time.time()
      session.lock.release()
      self.close_connection = True

  def end_headers(self):
    self.send_header("cache-control", "no-store")
    super().end_headers()

  def translate_path(self, path):
    root = Handler.root.resolve()
    target = (root / urlparse(path).path.lstrip("/")).resolve()
    return str(target if target == root or root in target.parents else root / "__not_found__")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--weights", default="weights/Qwen3.5-9B-UD-Q4_K_XL.gguf")
  parser.add_argument("--mtp-weights", default="weights/Qwen3.5-9B-UD-Q4_K_XL-MTP.gguf")
  parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
  parser.add_argument("--drafter", choices=("none", "mtp", "dflash"), default="none")
  parser.add_argument("--draft-weights", default="weights/qwen35-9b-dflash-Q4_K_M.gguf")
  parser.add_argument("--max-context", type=int, default=65536)
  parser.add_argument("--thinking", action="store_true")
  parser.add_argument("--speculative", action="store_true")
  parser.add_argument("--draft-tokens", type=int)
  parser.add_argument("--temperature", type=float)
  parser.add_argument("--top-p", type=float)
  parser.add_argument("--top-k", type=int, default=20)
  parser.add_argument("--preload", action="store_true")
  Handler.args, Handler.root = parser.parse_args(), Path(__file__).parent
  Handler.args.temperature = Handler.args.temperature if Handler.args.temperature is not None else 0.6 if Handler.args.thinking else 0.7
  Handler.args.top_p = Handler.args.top_p if Handler.args.top_p is not None else 0.95 if Handler.args.thinking else 0.8
  server = ThreadingHTTPServer((Handler.args.host, Handler.args.port), Handler)
  server.daemon_threads = True
  print(f"http://{Handler.args.host}:{Handler.args.port}")
  if Handler.args.preload: Handler.start_model_load(Handler.args.drafter)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    server.server_close()
    for session in list(Handler.sessions.values()):
      if session.runtime: session.runtime.close()
    if Handler.engine: Handler.engine.close()


if __name__ == "__main__": main()
