#!/usr/bin/env python3
"""Web chat UI — NVIDIA NIM + agent mode con terminal."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
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


def needs_thinking_kwargs(model: str) -> bool:
    m = model.lower()
    return "v4" in m or "thinking" in m or "kimi-k2" in m


async def execute_tool(name: str, args: dict) -> str:
    if name == "run_terminal":
        cmd = args.get("command", "echo 'sin comando'")
        cwd = str(Path(args.get("cwd", "~")).expanduser())
        timeout = min(int(args.get("timeout", 30)), 120)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
            )
            out = (result.stdout or "").rstrip()
            err = (result.stderr or "").rstrip()
            parts = []
            if out:
                parts.append(out)
            if err:
                parts.append(f"[stderr]\n{err}")
            return "\n".join(parts) or "(sin salida)"
        except subprocess.TimeoutExpired:
            return f"[timeout: proceso terminado después de {timeout}s]"
        except Exception as exc:
            return f"[error ejecutando comando: {exc}]"

    if name == "read_file":
        path = Path(args.get("path", "")).expanduser()
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if len(lines) > 500:
                return "\n".join(lines[:500]) + f"\n\n... ({len(lines) - 500} líneas más truncadas)"
            return content
        except Exception as exc:
            return f"[error leyendo archivo: {exc}]"

    if name == "write_file":
        path = Path(args.get("path", "")).expanduser()
        content = args.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"✓ Archivo guardado: {path} ({len(content.encode())} bytes)"
        except Exception as exc:
            return f"[error escribiendo archivo: {exc}]"

    if name == "list_directory":
        path = Path(args.get("path", "~")).expanduser()
        try:
            result = subprocess.run(
                f"ls -lah {path}", shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout or result.stderr or "(vacío)"
        except Exception as exc:
            return f"[error: {exc}]"

    return f"[herramienta '{name}' desconocida]"


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_response(
    messages: list,
    model: str,
    agent_mode: bool,
    system_prompt: str,
    temperature: float,
) -> AsyncGenerator[str, None]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    base_messages: list = []
    if system_prompt.strip():
        base_messages.append({"role": "system", "content": system_prompt.strip()})
    base_messages.extend(messages)

    current_messages = list(base_messages)
    max_iterations = 30

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
        for iteration in range(max_iterations):
            payload: dict = {
                "model": model,
                "messages": current_messages,
                "max_tokens": 32768,
                "temperature": temperature,
                "stream": True,
            }
            if needs_thinking_kwargs(model):
                payload["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
            if agent_mode:
                payload["tools"] = TOOLS
                payload["tool_choice"] = "auto"

            full_text = ""
            full_thinking = ""
            tool_calls_map: dict[int, dict] = {}
            finish_reason = None

            try:
                async with client.stream(
                    "POST", f"{BASE_URL}/chat/completions",
                    headers=headers, json=payload,
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield sse({"type": "error", "message": body.decode(errors="replace")})
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
                        thinking_delta = delta.get("reasoning_content") or ""
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
                yield sse({"type": "error", "message": "Timeout esperando respuesta del modelo (>300s)"})
                return
            except Exception as exc:
                yield sse({"type": "error", "message": str(exc)})
                return

            # No tool calls → done
            if not tool_calls_map or not agent_mode:
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

                yield sse({"type": "tool_call", "call_id": tc["id"], "name": name, "args": args})

                result = await execute_tool(name, args)

                yield sse({"type": "tool_result", "call_id": tc["id"], "name": name, "output": result})

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
        else:
            yield sse({"type": "error", "message": "Límite de iteraciones agente alcanzado (12)"})

    yield sse({"type": "done"})


@app.get("/api/models")
async def api_models():
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
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


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    return StreamingResponse(
        stream_response(
            messages=body.get("messages", []),
            model=body.get("model", "deepseek-ai/deepseek-v4-flash"),
            agent_mode=body.get("agent_mode", False),
            system_prompt=body.get("system_prompt", ""),
            temperature=float(body.get("temperature", 0.3)),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
