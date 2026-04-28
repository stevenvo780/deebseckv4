#!/usr/bin/env python3
"""Minimal NVIDIA Build/NIM API client.

The script intentionally avoids third-party dependencies so it can run in a
fresh checkout. It never prints the API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
RATE_HEADER_PREFIXES = ("x-ratelimit", "ratelimit", "retry-after")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def api_key() -> str | None:
    return (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_NIM_API_KEY")
        or os.getenv("NGC_API_KEY")
    )


def base_url() -> str:
    return os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def request_json(
    method: str,
    path: str,
    payload: dict | None = None,
    require_key: bool = False,
) -> tuple[int, dict[str, str], object]:
    key = api_key()
    if require_key and not key:
        raise SystemExit(
            "Falta NVIDIA_API_KEY. Crea .env desde .env.example o exporta la variable."
        )

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(
        f"{base_url()}{path}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            body = res.read().decode("utf-8", errors="replace")
            return res.status, dict(res.headers.items()), json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: object = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return exc.code, dict(exc.headers.items()), parsed


def selected_headers(headers: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in headers.items():
        lower = key.lower()
        if any(lower.startswith(prefix) for prefix in RATE_HEADER_PREFIXES):
            result[key] = value
    return result


def cmd_models(args: argparse.Namespace) -> int:
    status, headers, body = request_json("GET", "/models")
    if status != 200:
        print(json.dumps({"status": status, "headers": selected_headers(headers), "body": body}, indent=2))
        return 1

    models = sorted({item["id"] for item in body.get("data", [])})
    if args.filter:
        needle = args.filter.lower()
        models = [model for model in models if needle in model.lower()]

    print(f"base_url: {base_url()}")
    print(f"total_models: {len(models)}")
    for model in models:
        print(model)
    return 0


def chat_payload(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "Responde breve y en español.",
            },
            {
                "role": "user",
                "content": args.prompt,
            },
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "stream": False,
    }


def print_chat_response(status: int, headers: dict[str, str], body: object) -> int:
    rate_headers = selected_headers(headers)
    if rate_headers:
        print("rate_headers:")
        print(json.dumps(rate_headers, indent=2, ensure_ascii=False))

    print(f"status: {status}")
    if status != 200:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 1

    choice = body.get("choices", [{}])[0] if isinstance(body, dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content") or choice.get("text") or ""
    usage = body.get("usage", {}) if isinstance(body, dict) else {}

    print("model_response:")
    print(content.strip())
    if usage:
        print("usage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    status, headers, body = request_json(
        "POST",
        "/chat/completions",
        payload=chat_payload(args),
        require_key=True,
    )
    return print_chat_response(status, headers, body)


def cmd_probe(args: argparse.Namespace) -> int:
    failures = 0
    for idx in range(1, args.requests + 1):
        status, headers, body = request_json(
            "POST",
            "/chat/completions",
            payload=chat_payload(args),
            require_key=True,
        )
        rate_headers = selected_headers(headers)
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        print(
            json.dumps(
                {
                    "request": idx,
                    "status": status,
                    "rate_headers": rate_headers,
                    "usage": usage,
                    "error": body if status != 200 else None,
                },
                ensure_ascii=False,
            )
        )
        if status != 200:
            failures += 1
        if idx < args.requests:
            time.sleep(args.interval)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cliente mínimo para NVIDIA Build/NIM.")
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser("models", help="Lista modelos visibles en /v1/models.")
    models.add_argument("--filter", help="Filtra por texto, por ejemplo deepseek o qwen.")
    models.set_defaults(func=cmd_models)

    chat = sub.add_parser("chat", help="Ejecuta un consumo mínimo de chat/completions.")
    chat.add_argument("--model", default=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL))
    chat.add_argument("--prompt", default="Di OK y nombra el modelo que estas ejecutando.")
    chat.add_argument("--max-tokens", type=int, default=64)
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--top-p", type=float, default=0.7)
    chat.set_defaults(func=cmd_chat)

    probe = sub.add_parser("probe", help="Prueba controlada de limites con pocas solicitudes.")
    probe.add_argument("--model", default=os.getenv("NVIDIA_MODEL", DEFAULT_MODEL))
    probe.add_argument("--prompt", default="Responde solo OK.")
    probe.add_argument("--requests", type=int, default=3)
    probe.add_argument("--interval", type=float, default=2.0)
    probe.add_argument("--max-tokens", type=int, default=8)
    probe.add_argument("--temperature", type=float, default=0.0)
    probe.add_argument("--top-p", type=float, default=1.0)
    probe.set_defaults(func=cmd_probe)

    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

