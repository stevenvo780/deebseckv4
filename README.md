# NVIDIA Build / NIM API smoke test

Cliente mínimo para probar los endpoints OpenAI-compatible de NVIDIA Build/NIM desde este repo.

## Configuración

1. Crea `.env` desde el ejemplo:

```bash
cp .env.example .env
```

2. Edita `.env` y pega tu token:

```bash
NVIDIA_API_KEY=nvapi-...
```

No subas `.env`: está en `.gitignore`.

## Comandos

Listar modelos visibles actualmente:

```bash
python3 scripts/nvidia_nim.py models
```

Filtrar modelos relevantes:

```bash
python3 scripts/nvidia_nim.py models --filter deepseek
python3 scripts/nvidia_nim.py models --filter qwen
python3 scripts/nvidia_nim.py models --filter nemotron
```

Consumo mínimo con DeepSeek V4 Flash:

```bash
python3 scripts/nvidia_nim.py chat \
  --model deepseek-ai/deepseek-v4-flash \
  --prompt "Responde solo: OK NVIDIA NIM funciona." \
  --max-tokens 32
```

Consumo mínimo con un modelo rápido:

```bash
python3 scripts/nvidia_nim.py chat \
  --model nvidia/nemotron-3-nano-30b-a3b \
  --prompt "Responde solo: OK." \
  --max-tokens 16
```

Prueba controlada de rate limit, pequeña por defecto:

```bash
python3 scripts/nvidia_nim.py probe \
  --model deepseek-ai/deepseek-v4-flash \
  --requests 3 \
  --interval 2 \
  --max-tokens 8
```

Si el endpoint devuelve cabeceras como `Retry-After` o `x-ratelimit-*`, el script las imprime.

## Endpoint

Base URL:

```text
https://integrate.api.nvidia.com/v1
```

Endpoint usado:

```text
POST /chat/completions
GET  /models
```

## Modelos importantes vistos en el endpoint público

El endpoint público `/v1/models` devolvió 132 modelos el 2026-04-28. Subconjunto útil:

```text
deepseek-ai/deepseek-v4-flash
deepseek-ai/deepseek-v4-pro
deepseek-ai/deepseek-v3.2
qwen/qwen3-coder-480b-a35b-instruct
qwen/qwen3.5-397b-a17b
qwen/qwen3.5-122b-a10b
moonshotai/kimi-k2.5
moonshotai/kimi-k2-thinking
z-ai/glm-5.1
z-ai/glm5
minimaxai/minimax-m2.7
stepfun-ai/step-3.5-flash
openai/gpt-oss-120b
openai/gpt-oss-20b
nvidia/nemotron-3-nano-30b-a3b
nvidia/nemotron-3-super-120b-a12b
nvidia/llama-3.3-nemotron-super-49b-v1.5
nvidia/llama-3.1-nemotron-ultra-253b-v1
meta/llama-4-maverick-17b-128e-instruct
mistralai/devstral-2-123b-instruct-2512
mistralai/codestral-22b-instruct-v0.1
```

## Limites observados/documentados

NVIDIA documenta que los servicios OpenAI-compatible suelen limitar:

- requests por minuto
- tokens por minuto
- tokens totales por request

En la documentación pública no encontré una tabla oficial única con RPM/TPM por modelo para Build. En foros recientes de NVIDIA, usuarios reportan límite por defecto de `40 RPM` y errores `429 Too Many Requests`; esto debe verificarse con tu propia cuenta usando `probe`, porque NVIDIA puede variar límites por cuenta, región, modelo o carga.

El uso por Developer Program es para prototipado, desarrollo, investigación y testing. Producción requiere NVIDIA AI Enterprise.

