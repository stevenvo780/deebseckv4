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

En `.env` solo hacen falta credenciales y, si quieres, la base URL/modelo por defecto. Los ajustes operativos del chat (throttle NVIDIA y carpeta de trabajo) se cambian desde la UI.

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

## UI local con chat agéntico

Arranca la interfaz web:

```bash
bash web/start.sh
```

Abre:

```text
http://localhost:8000
```

En la barra lateral quedan solo el **prompt del sistema**, el **modelo** y el **historial**. Los ajustes operativos viven en el botón **Ajustes** de la barra superior.

La **carpeta de trabajo** configurada en ese modal se usa como `cwd` por defecto: comandos sin `cwd` explícito se ejecutan ahí, y las rutas relativas de `read_file`, `write_file` y `list_directory` se resuelven desde esa carpeta.

El chat funciona siempre en modo agéntico. Cada respuesta guarda y vuelve a mostrar:

- razonamiento emitido por el modelo
- comandos y llamadas a herramientas con su `cwd` o ruta resuelta
- resultados de herramientas y salidas completas (`stdout` / `stderr` cuando aplique)
- registro de errores y `stderr`
- memoria contextual automática entre turnos

Cada bloque de herramienta también muestra un botón **Ejecutar** para relanzar manualmente esa acción y ver una nueva salida dentro del mismo chat.

La memoria contextual se alimenta con tres capas a la vez:

- **memoria persistente por conversación**: resume objetivos, comandos, rutas, hallazgos y errores ya observados
- **memoria operativa reciente**: arrastra un resumen corto de los últimos turnos y acciones ejecutadas
- **salidas relevantes recientes**: reinyecta fragmentos útiles de outputs anteriores para mantener continuidad técnica

La temperatura ya no se configura manualmente desde la UI: el backend la elige automáticamente según el tipo de tarea y el modelo.

Además, el backend aplica un **throttle global** a las llamadas contra NVIDIA para mantenerse por debajo del límite de RPM. Por defecto usa `35 RPM`, pero también fuerza una **separación mínima entre requests** de `2.2s`, así que no dispara ráfagas cortas que suelen acabar en `429`. Si aun así NVIDIA responde `429`, entra en **backoff automático** y reintenta sin abortar la conversación.

Ese throttle también es **configurable desde el modal de Ajustes** de la UI con dos inputs:

- **RPM máximo**
- **intervalo mínimo entre requests (s)**

Y además puedes activar **modo de reintentos infinitos**, que elimina el tope normal de 30 iteraciones del agente y hace que siga reintentando fallos transitorios del proveedor en lugar de cerrar la conversación.

La **carpeta de trabajo** del modal también se guarda desde la UI. Estos ajustes quedan persistidos por la aplicación en disco, sin tener que editar `.env`.

La separación efectiva aplicada será siempre la mayor entre:

- `$60 / RPM$`
- el intervalo mínimo que configures manualmente

Esto afecta a las llamadas del chat y también a la carga de modelos (`/v1/models`).

La UI también guarda el historial en disco. Por defecto usa:

```text
.nimchat/conversations/*.json
```

Y los ajustes operativos de la UI se guardan en:

```text
.nimchat/settings.json
```

Puedes cambiar esa ubicación con:

```bash
NIMCHAT_DATA_DIR=/ruta/segura bash web/start.sh
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
