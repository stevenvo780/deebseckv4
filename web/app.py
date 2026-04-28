#!/usr/bin/env python3
"""Web chat UI — NVIDIA NIM + agent mode con terminal."""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"\''))


load_dotenv(Path(__file__).parent.parent / ".env")

API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
NIM_MAX_RPM = 35
NIM_MIN_REQUEST_INTERVAL_SECONDS = 2.2
NIM_RATE_WINDOW_SECONDS = 60.0
NIM_RATE_SAFETY_SECONDS = 0.35
NIM_429_BACKOFF_SECONDS = 15.0
DEFAULT_WORKING_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = Path(os.getenv("NIMCHAT_DATA_DIR", DEFAULT_WORKING_DIR / ".nimchat")).expanduser()
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_AGENT_MODE = True

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Ejecuta un comando shell en la terminal del PC del usuario. Úsalo para tareas del sistema, instalar paquetes, listar archivos, ejecutar scripts, buscar contenido, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando shell completo a ejecutar"},
                    "cwd": {"type": "string", "description": "Directorio de trabajo (opcional, default: home)"},
                    "timeout": {"type": "integer", "description": "Timeout en segundos, máximo 120", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lee el contenido completo de un archivo del sistema de archivos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta absoluta o relativa del archivo"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Escribe o sobreescribe contenido en un archivo. Crea directorios intermedios si no existen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del archivo a escribir"},
                    "content": {"type": "string", "description": "Contenido a escribir"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lista el contenido de un directorio con detalles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta del directorio", "default": "~"},
                },
                "required": [],
            },
        },
    },
]

app = FastAPI()


class NvidiaRequestRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        safety_seconds: float = 0.0,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1.0, float(window_seconds))
        self.safety_seconds = max(0.0, float(safety_seconds))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._timestamps: deque[float] = deque()
        self._blocked_until = 0.0
        self._last_request_at = 0.0
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def _effective_min_interval_seconds(self) -> float:
        return max(self.window_seconds / self.max_requests, self.min_interval_seconds)

    def snapshot(self) -> dict[str, float]:
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "safety_seconds": self.safety_seconds,
            "min_interval_seconds": self.min_interval_seconds,
            "effective_min_interval_seconds": self._effective_min_interval_seconds(),
        }

    async def update_settings(
        self,
        max_requests: int | None = None,
        min_interval_seconds: float | None = None,
    ) -> dict[str, float]:
        async with self._lock:
            if max_requests is not None:
                self.max_requests = max(1, int(max_requests))
            if min_interval_seconds is not None:
                self.min_interval_seconds = max(0.0, float(min_interval_seconds))
            self._prune(time.monotonic())
            return self.snapshot()

    async def acquire(self) -> dict[str, Any]:
        waited = 0.0
        reasons: set[str] = set()
        while True:
            delay = 0.0
            async with self._lock:
                now = time.monotonic()
                self._prune(now)

                if self._blocked_until > now:
                    delay = max(delay, self._blocked_until - now)
                    reasons.add("backoff")

                effective_min_interval = self._effective_min_interval_seconds()
                if self._last_request_at > 0 and effective_min_interval > 0:
                    gap_delay = effective_min_interval - (now - self._last_request_at)
                    if gap_delay > 0:
                        delay = max(delay, gap_delay)
                        reasons.add("spacing")

                if len(self._timestamps) >= self.max_requests:
                    oldest = self._timestamps[0]
                    delay = max(delay, self.window_seconds - (now - oldest) + self.safety_seconds)
                    reasons.add("rpm")

                if delay <= 0:
                    stamp = time.monotonic()
                    self._prune(stamp)
                    self._timestamps.append(stamp)
                    self._last_request_at = stamp
                    return {
                        "waited": waited,
                        "reasons": sorted(reasons),
                        **self.snapshot(),
                    }

            await asyncio.sleep(delay)
            waited += delay

    async def impose_cooldown(self, seconds: float) -> float:
        extra = max(0.0, float(seconds))
        if extra <= 0:
            return 0.0

        cooldown = extra + self.safety_seconds
        async with self._lock:
            now = time.monotonic()
            self._blocked_until = max(self._blocked_until, now + cooldown)
        return cooldown


NVIDIA_RATE_LIMITER = NvidiaRequestRateLimiter(
    max_requests=NIM_MAX_RPM,
    window_seconds=NIM_RATE_WINDOW_SECONDS,
    safety_seconds=NIM_RATE_SAFETY_SECONDS,
    min_interval_seconds=NIM_MIN_REQUEST_INTERVAL_SECONDS,
)


def needs_thinking_kwargs(model: str) -> bool:
    m = model.lower()
    return "v4" in m or "thinking" in m or "kimi-k2" in m


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    value = max(0.0, float(seconds))
    if value >= 10:
        return f"{value:.0f}s"
    if value >= 1:
        return f"{value:.1f}s"
    return f"{value:.2f}s"


def parse_retry_after_value(raw: str | None) -> float | None:
    if not raw:
        return None

    text = str(raw).strip()
    if not text:
        return None

    try:
        numeric = float(text)
    except ValueError:
        numeric = None

    if numeric is not None:
        if numeric > 1_000_000_000:
            return max(0.0, numeric - time.time())
        return max(0.0, numeric)

    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, retry_at.timestamp() - time.time())


def resolve_retry_after_seconds(headers: httpx.Headers) -> float:
    for header in (
        "retry-after",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ):
        parsed = parse_retry_after_value(headers.get(header))
        if parsed and parsed > 0:
            return parsed
    return NIM_429_BACKOFF_SECONDS


def parse_runtime_config_value(value: Any, *, kind: str, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Valor invalido para {kind}") from exc
    if parsed < minimum:
        raise HTTPException(status_code=400, detail=f"{kind} debe ser >= {minimum}")
    return parsed


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def ensure_conversations_dir() -> None:
    ensure_data_dir()
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_working_dir_value(value: str | None = None) -> str:
    raw = (value or str(DEFAULT_WORKING_DIR)).strip() or str(DEFAULT_WORKING_DIR)
    return str(Path(raw).expanduser().resolve(strict=False))


def default_runtime_settings() -> dict[str, Any]:
    return {
        "working_dir": normalize_working_dir_value(str(DEFAULT_WORKING_DIR)),
        "nvidia_max_rpm": int(NIM_MAX_RPM),
        "nvidia_min_request_interval_seconds": float(NIM_MIN_REQUEST_INTERVAL_SECONDS),
    }


def sanitize_runtime_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    settings = default_runtime_settings()
    if not isinstance(raw, dict):
        return settings

    working_dir = raw.get("working_dir")
    if isinstance(working_dir, str) and working_dir.strip():
        settings["working_dir"] = normalize_working_dir_value(working_dir)

    try:
        rpm = int(float(raw.get("nvidia_max_rpm")))
        if rpm >= 1:
            settings["nvidia_max_rpm"] = rpm
    except (TypeError, ValueError):
        pass

    try:
        min_interval = float(raw.get("nvidia_min_request_interval_seconds"))
        if min_interval >= 0:
            settings["nvidia_min_request_interval_seconds"] = min_interval
    except (TypeError, ValueError):
        pass

    return settings


def read_runtime_settings() -> dict[str, Any]:
    ensure_data_dir()
    if not SETTINGS_FILE.exists():
        return default_runtime_settings()

    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_runtime_settings()

    return sanitize_runtime_settings(raw)


def write_runtime_settings(settings: dict[str, Any]) -> dict[str, Any]:
    ensure_data_dir()
    sanitized = sanitize_runtime_settings(settings)
    tmp_path = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(SETTINGS_FILE)
    return sanitized


INITIAL_RUNTIME_SETTINGS = read_runtime_settings()
NVIDIA_RATE_LIMITER.max_requests = int(INITIAL_RUNTIME_SETTINGS["nvidia_max_rpm"])
NVIDIA_RATE_LIMITER.min_interval_seconds = float(INITIAL_RUNTIME_SETTINGS["nvidia_min_request_interval_seconds"])


def current_runtime_settings() -> dict[str, Any]:
    stored = read_runtime_settings()
    limiter = NVIDIA_RATE_LIMITER.snapshot()
    return {
        "working_dir": stored["working_dir"],
        "nvidia_max_rpm": int(limiter["max_requests"]),
        "nvidia_min_request_interval_seconds": float(limiter["min_interval_seconds"]),
        "nvidia_effective_min_interval_seconds": float(limiter["effective_min_interval_seconds"]),
        "nvidia_rate_window_seconds": float(limiter["window_seconds"]),
        "nvidia_rate_safety_seconds": float(limiter["safety_seconds"]),
        "nvidia_429_backoff_seconds": float(NIM_429_BACKOFF_SECONDS),
    }


def sanitize_chat_messages(messages: list[Any]) -> list[dict[str, Any]]:
    clean_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue

        clean_message: dict[str, Any] = {"role": role}

        if role == "tool":
            clean_message["tool_call_id"] = message.get("tool_call_id")
            clean_message["content"] = message.get("content") or ""
        else:
            content = message.get("content")
            if role == "assistant" and not content and not message.get("tool_calls"):
                continue
            clean_message["content"] = content or ""
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                clean_message["tool_calls"] = message["tool_calls"]

        clean_messages.append({k: v for k, v in clean_message.items() if v is not None})

    return clean_messages


def stringify_reasoning_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(stringify_reasoning_fragment(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "thinking", "value"):
            if key in value:
                return stringify_reasoning_fragment(value[key])
    return ""


def extract_reasoning_delta(delta: dict[str, Any]) -> str:
    parts: list[str] = []

    for key in ("reasoning_content", "reasoning", "thinking"):
        fragment = stringify_reasoning_fragment(delta.get(key))
        if fragment:
            parts.append(fragment)

    content = delta.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") in {"reasoning", "reasoning_content", "thinking"}:
                fragment = stringify_reasoning_fragment(item)
                if fragment:
                    parts.append(fragment)

    return "".join(parts)


def tool_output_flags(output: str) -> dict[str, bool]:
    normalized = output.strip().lower()
    has_stderr = "[stderr]" in normalized
    has_error = (
        normalized.startswith("[error")
        or normalized.startswith("[timeout")
        or has_stderr
    )
    return {
        "has_error": has_error,
        "has_stderr": has_stderr,
    }


def provider_tools_degraded(body_text: str) -> bool:
    normalized = body_text.lower()
    return "degraded function cannot be invoked" in normalized


def latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "") or "")
    return ""


def choose_temperature(model: str, messages: list[dict[str, Any]], system_prompt: str) -> float:
    model_name = model.lower()
    prompt = f"{system_prompt}\n{latest_user_message(messages)}".lower()

    coding_hints = (
        "code", "código", "script", "debug", "error", "traceback", "fix", "corrige",
        "terminal", "bash", "shell", "python", "sql", "api", "json", "regex",
    )
    analysis_hints = (
        "explica", "explain", "analiza", "analyze", "resume", "summary", "compara",
        "compare", "arquitectura", "architecture", "documenta", "document",
    )
    creative_hints = (
        "poema", "cuento", "historia", "story", "marketing", "copy", "creativo", "creative",
    )

    if any(token in model_name for token in ("coder", "codestral", "devstral")):
        return 0.15
    if any(token in prompt for token in coding_hints):
        return 0.15
    if any(token in prompt for token in creative_hints):
        return 0.65
    if any(token in prompt for token in analysis_hints):
        return 0.30
    if needs_thinking_kwargs(model):
        return 0.22
    return 0.28


def build_tool_preview(name: str, args: dict[str, Any], working_dir: Path) -> dict[str, Any]:
    if name == "run_terminal":
        return {
            "command": args.get("command", ""),
            "cwd": str(resolve_path(args.get("cwd") or ".", working_dir)),
        }
    if name in {"read_file", "write_file"}:
        return {
            "path": str(resolve_path(args.get("path"), working_dir)),
        }
    if name == "list_directory":
        return {
            "path": str(resolve_path(args.get("path") or ".", working_dir)),
        }
    return {}


def conversation_path(conversation_id: str) -> Path:
    if not conversation_id or any(not (c.isalnum() or c in "-_") for c in conversation_id):
        raise HTTPException(status_code=400, detail="conversation_id invalido")
    return CONVERSATIONS_DIR / f"{conversation_id}.json"


def read_conversation_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Conversacion no encontrada") from None
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Conversacion corrupta: {path.name}") from exc


def conversation_title(messages: list, fallback: str | None = None) -> str:
    if fallback and fallback.strip():
        return fallback.strip()[:80]
    for message in messages:
        if message.get("role") == "user":
            text = " ".join(str(message.get("content", "")).split())
            if text:
                return text[:80]
    return "Nueva conversacion"


def conversation_metadata(doc: dict) -> dict:
    messages = doc.get("messages") if isinstance(doc.get("messages"), list) else []
    last_message = ""
    for message in reversed(messages):
        content = str(message.get("content", "")).strip()
        if content:
            last_message = " ".join(content.split())[:120]
            break
    return {
        "id": doc.get("id"),
        "title": doc.get("title") or conversation_title(messages),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
        "model": doc.get("model"),
        "working_dir": doc.get("working_dir"),
        "message_count": len(messages),
        "last_message": last_message,
    }


def write_conversation(doc: dict) -> None:
    ensure_conversations_dir()
    path = conversation_path(doc["id"])
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def resolve_working_dir(value: str | None = None) -> Path:
    configured = read_runtime_settings().get("working_dir") or str(DEFAULT_WORKING_DIR)
    raw = (value or str(configured)).strip() or str(DEFAULT_WORKING_DIR)
    return Path(raw).expanduser().resolve(strict=False)


def resolve_path(path_value: str | None, working_dir: Path) -> Path:
    raw = (path_value or ".").strip() or "."
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = working_dir / path
    return path.resolve(strict=False)


def compact_inline_text(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def compact_block_text(value: Any, max_chars: int = 700, max_lines: int = 18) -> str:
    lines = str(value or "").strip().splitlines()
    if not lines:
        return ""

    block = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        block += "\n..."

    if len(block) <= max_chars:
        return block
    return block[: max_chars - 1].rstrip() + "…"


def take_recent_unique(items: list[Any], key_fn, limit: int) -> list[Any]:
    picked: list[Any] = []
    seen: set[str] = set()
    for item in reversed(items):
        key = str(key_fn(item) or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        picked.append(item)
        if len(picked) >= limit:
            break
    picked.reverse()
    return picked


def extract_trace_events(messages: list[Any]) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue

        trace = message.get("trace")
        if not isinstance(trace, dict):
            continue

        tool_calls: dict[str, dict[str, Any]] = {}
        for raw_event in trace.get("events") or []:
            if not isinstance(raw_event, dict):
                continue

            event = dict(raw_event)
            if event.get("type") == "tool_call" and event.get("call_id"):
                tool_calls[str(event["call_id"])] = event
            elif event.get("type") == "tool_result":
                call = tool_calls.get(str(event.get("call_id") or ""))
                if call:
                    for key in ("command", "cwd", "path", "args", "name"):
                        if key not in event and key in call:
                            event[key] = call[key]

            collected.append(event)

    return collected


def build_context_memory(messages: list[Any]) -> dict[str, Any]:
    user_goals = [
        compact_inline_text(message.get("content"), 220)
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and str(message.get("content") or "").strip()
    ]
    assistant_findings = [
        compact_inline_text(message.get("content"), 320)
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "assistant"
        and str(message.get("content") or "").strip()
    ]

    commands: list[dict[str, str]] = []
    files: list[dict[str, str]] = []
    tool_findings: list[str] = []
    errors: list[str] = []

    for event in extract_trace_events(messages):
        event_type = event.get("type")
        name = compact_inline_text(event.get("name") or "tool", 40)

        if event_type == "tool_call":
            command = str(event.get("command") or "").strip()
            if name == "run_terminal" and command:
                commands.append({
                    "command": compact_inline_text(command, 180),
                    "cwd": compact_inline_text(event.get("cwd"), 140),
                })

            path = str(event.get("path") or "").strip()
            if path:
                files.append({
                    "action": name,
                    "path": compact_inline_text(path, 200),
                })

        elif event_type == "tool_result":
            path = str(event.get("path") or "").strip()
            if path:
                files.append({
                    "action": name,
                    "path": compact_inline_text(path, 200),
                })

            output = compact_block_text(event.get("output"), 320, 8)
            if event.get("has_error"):
                errors.append(f"{name}: {compact_inline_text(output, 220)}")
            elif output and name in {"run_terminal", "read_file", "list_directory", "write_file"}:
                tool_findings.append(f"{name}: {compact_inline_text(output, 220)}")

        elif event_type == "error":
            source = compact_inline_text(event.get("source") or "error", 40)
            message = compact_inline_text(event.get("message"), 220)
            if message:
                errors.append(f"{source}: {message}")

    return {
        "objectives": take_recent_unique(user_goals, lambda item: item, 4),
        "assistant_findings": take_recent_unique(assistant_findings, lambda item: item, 3),
        "commands": take_recent_unique(commands, lambda item: f"{item.get('command')}|{item.get('cwd')}", 5),
        "files": take_recent_unique(files, lambda item: f"{item.get('action')}|{item.get('path')}", 8),
        "tool_findings": take_recent_unique(tool_findings, lambda item: item, 4),
        "errors": take_recent_unique(errors, lambda item: item, 5),
    }


def render_context_memory(memory: dict[str, Any] | None) -> str:
    if not isinstance(memory, dict):
        return ""

    sections: list[str] = []

    objectives = memory.get("objectives") or []
    if objectives:
        sections.append("Objetivos recientes del usuario:\n" + "\n".join(f"- {item}" for item in objectives))

    assistant_findings = memory.get("assistant_findings") or []
    if assistant_findings:
        sections.append("Conclusiones recientes ya dadas:\n" + "\n".join(f"- {item}" for item in assistant_findings))

    commands = memory.get("commands") or []
    if commands:
        sections.append(
            "Comandos ya ejecutados y observados:\n"
            + "\n".join(
                f"- {item.get('command')}" + (f" (cwd: {item.get('cwd')})" if item.get("cwd") else "")
                for item in commands
            )
        )

    files = memory.get("files") or []
    if files:
        sections.append(
            "Rutas ya inspeccionadas o tocadas:\n"
            + "\n".join(f"- {item.get('action')}: {item.get('path')}" for item in files)
        )

    tool_findings = memory.get("tool_findings") or []
    if tool_findings:
        sections.append("Hallazgos técnicos recientes:\n" + "\n".join(f"- {item}" for item in tool_findings))

    errors = memory.get("errors") or []
    if errors:
        sections.append("Errores o fricciones recientes:\n" + "\n".join(f"- {item}" for item in errors))

    if not sections:
        return ""

    return (
        "Memoria persistente de la conversación. Mantén continuidad con estos hechos ya observados, "
        "sin inventar detalles no confirmados.\n\n"
        + "\n\n".join(sections)
    )


def build_recent_activity_summary(messages: list[Any]) -> str:
    recent_events = extract_trace_events(messages)[-10:]
    lines: list[str] = []

    for event in recent_events:
        event_type = event.get("type")
        name = compact_inline_text(event.get("name") or "tool", 40)

        if event_type == "tool_call":
            if name == "run_terminal" and str(event.get("command") or "").strip():
                lines.append(
                    f"- run_terminal: {compact_inline_text(event.get('command'), 160)}"
                    + (f" (cwd: {compact_inline_text(event.get('cwd'), 100)})" if event.get("cwd") else "")
                )
            elif event.get("path"):
                lines.append(f"- {name}: {compact_inline_text(event.get('path'), 180)}")

        elif event_type == "tool_result" and event.get("has_error"):
            lines.append(f"- error en {name}: {compact_inline_text(event.get('output'), 180)}")

        elif event_type == "error":
            lines.append(f"- error general: {compact_inline_text(event.get('message'), 180)}")

    if not lines:
        return ""

    return "Resumen operativo reciente (últimos turnos):\n" + "\n".join(lines[-8:])


def build_relevant_tool_outputs(messages: list[Any]) -> str:
    relevant_events = [
        event
        for event in extract_trace_events(messages)
        if event.get("type") == "tool_result"
        and str(event.get("output") or "").strip()
        and (
            event.get("has_error")
            or event.get("name") in {"run_terminal", "read_file", "list_directory"}
        )
    ]
    selected = take_recent_unique(
        relevant_events,
        lambda event: event.get("call_id") or f"{event.get('name')}|{event.get('path')}|{event.get('command')}",
        4,
    )

    if not selected:
        return ""

    blocks: list[str] = []
    for index, event in enumerate(selected, start=1):
        header_parts = [compact_inline_text(event.get("name") or "tool", 40)]
        if event.get("command"):
            header_parts.append(f"comando: {compact_inline_text(event.get('command'), 140)}")
        if event.get("path"):
            header_parts.append(f"ruta: {compact_inline_text(event.get('path'), 140)}")
        if event.get("cwd"):
            header_parts.append(f"cwd: {compact_inline_text(event.get('cwd'), 120)}")
        if isinstance(event.get("exit_code"), int):
            header_parts.append(f"exit {event.get('exit_code')}")
        if event.get("timed_out"):
            header_parts.append("timeout")

        output = compact_block_text(event.get("output"), 700, 12)
        blocks.append(
            f"{index}. {' · '.join(header_parts)}\n"
            f"Salida observada:\n{output}"
        )

    return (
        "Fragmentos recientes de salidas relevantes. Úsalos como hechos observados si siguen siendo útiles:\n\n"
        + "\n\n".join(blocks)
    )


async def execute_tool(name: str, args: dict, working_dir: Path) -> tuple[str, dict[str, Any]]:
    if name == "run_terminal":
        cmd = args.get("command", "echo 'sin comando'")
        cwd = resolve_path(args.get("cwd") or ".", working_dir)
        timeout = min(int(args.get("timeout", 30)), 120)
        metadata = {
            "command": cmd,
            "cwd": str(cwd),
            "timeout": timeout,
        }
        try:
            env = os.environ.copy()
            env["PWD"] = str(cwd)
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(cwd), env=env,
            )
            metadata["exit_code"] = result.returncode
            out = (result.stdout or "").rstrip()
            err = (result.stderr or "").rstrip()
            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr]\n{err}")
            return "\n".join(parts) or "(sin salida)", metadata
        except subprocess.TimeoutExpired:
            metadata["timed_out"] = True
            return f"[timeout: proceso terminado después de {timeout}s]", metadata
        except Exception as exc:
            metadata["failed"] = True
            return f"[error ejecutando comando: {exc}]", metadata

    if name == "read_file":
        path = resolve_path(args.get("path"), working_dir)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > 500:
                return "\n".join(lines[:500]) + f"\n\n... ({len(lines) - 500} líneas más truncadas)", {"path": str(path), "line_count": len(lines)}
            return content, {"path": str(path), "line_count": len(lines)}
        except Exception as exc:
            return f"[error leyendo archivo: {exc}]", {"path": str(path), "failed": True}

    if name == "write_file":
        path = resolve_path(args.get("path"), working_dir)
        content = args.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"✓ Archivo guardado: {path} ({len(content.encode())} bytes)", {"path": str(path), "bytes_written": len(content.encode())}
        except Exception as exc:
            return f"[error escribiendo archivo: {exc}]", {"path": str(path), "failed": True}

    if name == "list_directory":
        path = resolve_path(args.get("path") or ".", working_dir)
        try:
            result = subprocess.run(
                ["ls", "-lah", str(path)], capture_output=True, text=True, timeout=10
            )
            return result.stdout or result.stderr or "(vacío)", {"path": str(path), "exit_code": result.returncode}
        except Exception as exc:
            return f"[error: {exc}]", {"path": str(path), "failed": True}

    return f"[herramienta '{name}' desconocida]", {"failed": True}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_response(
    messages: list,
    model: str,
    agent_mode: bool,
    system_prompt: str,
    temperature: float | None,
    working_dir: Path,
    conversation_memory: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    sanitized_messages = sanitize_chat_messages(messages)
    selected_temperature = choose_temperature(model, sanitized_messages, system_prompt)
    persistent_memory_text = render_context_memory(conversation_memory or build_context_memory(messages))
    recent_activity_text = build_recent_activity_summary(messages)
    relevant_outputs_text = build_relevant_tool_outputs(messages)

    base_messages: list = []
    if agent_mode:
        base_messages.append({
            "role": "system",
            "content": (
                f"Directorio de trabajo seleccionado: {working_dir}. "
                "Interpreta rutas relativas desde ese directorio. "
                "Si ejecutas comandos con run_terminal y no necesitas otra ruta, omite cwd."
            ),
        })
    if system_prompt.strip():
        base_messages.append({"role": "system", "content": system_prompt.strip()})
    if persistent_memory_text:
        base_messages.append({"role": "system", "content": persistent_memory_text})
    if recent_activity_text:
        base_messages.append({"role": "system", "content": recent_activity_text})
    if relevant_outputs_text:
        base_messages.append({"role": "system", "content": relevant_outputs_text})
    base_messages.extend(sanitized_messages)

    current_messages = list(base_messages)
    max_iterations = 30
    tools_temporarily_disabled = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
        for iteration in range(max_iterations):
            effective_agent_mode = agent_mode and not tools_temporarily_disabled
            payload: dict = {
                "model": model,
                "messages": current_messages,
                "max_tokens": 32768,
                "temperature": selected_temperature,
                "stream": True,
            }
            if needs_thinking_kwargs(model):
                payload["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
            if effective_agent_mode:
                payload["tools"] = TOOLS
                payload["tool_choice"] = "auto"

            full_text = ""
            full_thinking = ""
            tool_calls_map: dict[int, dict] = {}
            finish_reason = None
            retry_without_tools = False

            while True:
                acquire_info = await NVIDIA_RATE_LIMITER.acquire()
                waited_for_slot = float(acquire_info.get("waited") or 0.0)
                if waited_for_slot >= 0.75:
                    reason_labels = {
                        "backoff": "backoff 429",
                        "rpm": "ventana RPM",
                        "spacing": "intervalo mínimo",
                    }
                    reasons = [reason_labels.get(reason, reason) for reason in acquire_info.get("reasons") or []]
                    yield sse({
                        "type": "notice",
                        "message": (
                            f"Throttle NVIDIA aplicado: espera de {format_duration(waited_for_slot)} "
                            f"para respetar el máximo configurado de {int(acquire_info.get('max_requests') or NIM_MAX_RPM)} RPM"
                            f" con una separación efectiva de {format_duration(acquire_info.get('effective_min_interval_seconds') or 0)} entre requests."
                        ),
                        "source": "rate_limit_queue",
                        "wait_seconds": round(waited_for_slot, 2),
                        "max_rpm": int(acquire_info.get("max_requests") or NIM_MAX_RPM),
                        "min_interval_seconds": round(float(acquire_info.get("effective_min_interval_seconds") or 0.0), 2),
                        "reasons": reasons,
                        "ts": utc_now(),
                    })

                try:
                    async with client.stream(
                        "POST", f"{BASE_URL}/chat/completions",
                        headers=headers, json=payload,
                    ) as resp:
                        if resp.status_code != 200:
                            body = await resp.aread()
                            body_text = body.decode(errors="replace")

                            if resp.status_code == 429:
                                backoff_seconds = resolve_retry_after_seconds(resp.headers)
                                cooldown = await NVIDIA_RATE_LIMITER.impose_cooldown(backoff_seconds)
                                yield sse({
                                    "type": "notice",
                                    "message": (
                                        "NVIDIA devolvió 429 Too Many Requests. "
                                        f"Entrando en backoff automático durante {format_duration(cooldown)} "
                                        f"y reintentando sin abortar la conversación."
                                    ),
                                    "source": "rate_limit_backoff",
                                    "status": resp.status_code,
                                    "wait_seconds": round(cooldown, 2),
                                    "max_rpm": NIM_MAX_RPM,
                                    "ts": utc_now(),
                                })
                                continue

                            if effective_agent_mode and provider_tools_degraded(body_text):
                                tools_temporarily_disabled = True
                                retry_without_tools = True
                                yield sse({
                                    "type": "error",
                                    "message": (
                                        "El proveedor rechazó temporalmente la invocación de herramientas; "
                                        "reintentando esta misma respuesta sin tools para no bloquear el chat.\n\n"
                                        f"{body_text}"
                                    ),
                                    "source": "agent_tools_unavailable",
                                    "status": resp.status_code,
                                    "ts": utc_now(),
                                })
                                break

                            yield sse({
                                "type": "error",
                                "message": body_text,
                                "source": "model_http",
                                "status": resp.status_code,
                                "ts": utc_now(),
                            })
                            return

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            choice = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason") or finish_reason

                            # Reasoning / thinking tokens
                            thinking_delta = extract_reasoning_delta(delta)
                            if thinking_delta:
                                full_thinking += thinking_delta
                                yield sse({"type": "thinking", "content": thinking_delta})

                            # Normal content
                            text_delta = delta.get("content") or ""
                            if text_delta:
                                full_text += text_delta
                                yield sse({"type": "text", "content": text_delta})

                            # Tool calls (streamed in chunks)
                            for tc_delta in delta.get("tool_calls") or []:
                                idx = tc_delta.get("index", 0)
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {"id": "", "name": "", "args_str": ""}
                                entry = tool_calls_map[idx]
                                if tc_delta.get("id"):
                                    entry["id"] = tc_delta["id"]
                                fn = tc_delta.get("function") or {}
                                if fn.get("name"):
                                    entry["name"] = fn["name"]
                                if fn.get("arguments"):
                                    entry["args_str"] += fn["arguments"]

                except httpx.ReadTimeout:
                    yield sse({
                        "type": "error",
                        "message": "Timeout esperando respuesta del modelo (>300s)",
                        "source": "model_timeout",
                        "ts": utc_now(),
                    })
                    return
                except Exception as exc:
                    yield sse({
                        "type": "error",
                        "message": str(exc),
                        "source": "model_exception",
                        "ts": utc_now(),
                    })
                    return

                break

            if retry_without_tools:
                continue

            # No tool calls → done
            if not tool_calls_map or not effective_agent_mode:
                break

            # Build assistant message with tool_calls for the next round
            tool_calls_list = []
            for idx in sorted(tool_calls_map.keys()):
                tc = tool_calls_map[idx]
                tool_calls_list.append({
                    "id": tc["id"] or f"call_{iteration}_{idx}",
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["args_str"]},
                })

            current_messages.append({
                "role": "assistant",
                "content": full_text or None,
                "tool_calls": tool_calls_list,
            })

            # Execute each tool and stream results
            for tc in tool_calls_list:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                preview = build_tool_preview(name, args, working_dir)

                yield sse({
                    "type": "tool_call",
                    "call_id": tc["id"],
                    "name": name,
                    "args": args,
                    "ts": utc_now(),
                    **preview,
                })

                result, metadata = await execute_tool(name, args, working_dir)
                flags = tool_output_flags(result)

                yield sse({
                    "type": "tool_result",
                    "call_id": tc["id"],
                    "name": name,
                    "output": result,
                    "ts": utc_now(),
                    **metadata,
                    **flags,
                })

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        else:
            yield sse({
                "type": "error",
                "message": "Límite de iteraciones agente alcanzado (30)",
                "source": "agent_limit",
                "ts": utc_now(),
            })

    yield sse({"type": "done"})


@app.get("/api/models")
async def api_models():
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            while True:
                await NVIDIA_RATE_LIMITER.acquire()
                resp = await client.get(
                    f"{BASE_URL}/models",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )

                if resp.status_code == 429:
                    cooldown = resolve_retry_after_seconds(resp.headers)
                    await NVIDIA_RATE_LIMITER.impose_cooldown(cooldown)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    all_ids = sorted({m["id"] for m in data.get("data", [])})
                    exclude_kw = [
                        "embed", "rerank", "reward", "clip", "guard", "safety",
                        "detector", "pii", "translate", "parse", "calibration",
                        "chatqa", "retriev", "nv-embed",
                    ]
                    filtered = [m for m in all_ids if not any(kw in m.lower() for kw in exclude_kw)]
                    return {"models": filtered, "total": len(filtered)}
                return {"models": [], "error": resp.text}
    except Exception as exc:
        return {"models": [], "error": str(exc)}


@app.get("/api/workdir")
async def api_workdir():
    runtime_settings = current_runtime_settings()
    return {
        "default": str(DEFAULT_WORKING_DIR),
        "home": str(Path.home()),
        "working_dir": runtime_settings["working_dir"],
        "default_agent_mode": DEFAULT_AGENT_MODE,
        "nvidia_max_rpm": runtime_settings["nvidia_max_rpm"],
        "nvidia_rate_window_seconds": runtime_settings["nvidia_rate_window_seconds"],
        "nvidia_min_request_interval_seconds": runtime_settings["nvidia_min_request_interval_seconds"],
        "nvidia_effective_min_interval_seconds": runtime_settings["nvidia_effective_min_interval_seconds"],
        "nvidia_rate_safety_seconds": runtime_settings["nvidia_rate_safety_seconds"],
        "nvidia_429_backoff_seconds": runtime_settings["nvidia_429_backoff_seconds"],
    }


@app.post("/api/settings")
async def api_settings(request: Request):
    body = await request.json()
    runtime_settings = read_runtime_settings()

    max_rpm = runtime_settings["nvidia_max_rpm"]
    if "nvidia_max_rpm" in body:
        max_rpm = int(parse_runtime_config_value(body.get("nvidia_max_rpm"), kind="nvidia_max_rpm", minimum=1.0))

    min_interval = runtime_settings["nvidia_min_request_interval_seconds"]
    if "nvidia_min_request_interval_seconds" in body:
        min_interval = parse_runtime_config_value(
            body.get("nvidia_min_request_interval_seconds"),
            kind="nvidia_min_request_interval_seconds",
            minimum=0.0,
        )

    if "working_dir" in body:
        runtime_settings["working_dir"] = normalize_working_dir_value(str(body.get("working_dir") or ""))

    runtime_settings["nvidia_max_rpm"] = max_rpm
    runtime_settings["nvidia_min_request_interval_seconds"] = min_interval
    saved_settings = write_runtime_settings(runtime_settings)

    await NVIDIA_RATE_LIMITER.update_settings(
        max_requests=int(saved_settings["nvidia_max_rpm"]),
        min_interval_seconds=float(saved_settings["nvidia_min_request_interval_seconds"]),
    )
    return {
        "settings": current_runtime_settings()
    }


@app.get("/api/conversations")
async def api_conversations():
    ensure_conversations_dir()
    items = []
    for path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            items.append(conversation_metadata(read_conversation_file(path)))
        except HTTPException:
            continue
    items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return {"conversations": items, "storage_dir": str(CONVERSATIONS_DIR)}


@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: str):
    return read_conversation_file(conversation_path(conversation_id))


@app.post("/api/conversations")
async def api_save_conversation(request: Request):
    body = await request.json()
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    conversation_id = body.get("id") or str(uuid.uuid4())
    existing = {}
    path = conversation_path(conversation_id)
    if path.exists():
        existing = read_conversation_file(path)

    now = utc_now()
    doc = {
        "id": conversation_id,
        "title": conversation_title(messages, body.get("title")),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "model": body.get("model"),
        "working_dir": str(resolve_working_dir(body.get("working_dir"))),
        "agent_mode": DEFAULT_AGENT_MODE,
        "system_prompt": body.get("system_prompt", ""),
        "messages": messages,
        "context_memory": build_context_memory(messages),
        "memory_updated_at": now,
    }
    write_conversation(doc)
    return {"conversation": doc, "metadata": conversation_metadata(doc)}


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    path = conversation_path(conversation_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversacion no encontrada")
    path.unlink()
    return {"deleted": conversation_id}


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    working_dir = resolve_working_dir(body.get("working_dir"))
    conversation_memory = None
    conversation_id = body.get("conversation_id")
    if conversation_id:
        try:
            conversation_doc = read_conversation_file(conversation_path(str(conversation_id)))
            if isinstance(conversation_doc.get("context_memory"), dict):
                conversation_memory = conversation_doc["context_memory"]
        except HTTPException:
            conversation_memory = None

    return StreamingResponse(
        stream_response(
            messages=body.get("messages", []),
            model=body.get("model", "deepseek-ai/deepseek-v4-flash"),
            agent_mode=DEFAULT_AGENT_MODE,
            system_prompt=body.get("system_prompt", ""),
            temperature=None,
            working_dir=working_dir,
            conversation_memory=conversation_memory,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tools/execute")
async def api_execute_tool(request: Request):
    body = await request.json()
    name = str(body.get("name") or "").strip()
    args = body.get("args") if isinstance(body.get("args"), dict) else {}
    working_dir = resolve_working_dir(body.get("working_dir"))

    output, metadata = await execute_tool(name, args, working_dir)
    return {
        "event": {
            "type": "tool_result",
            "call_id": body.get("call_id") or f"manual_{uuid.uuid4().hex[:8]}",
            "name": name,
            "output": output,
            "ts": utc_now(),
            **metadata,
            **tool_output_flags(output),
        }
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
