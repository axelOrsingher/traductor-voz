# voztrad — traductor de voz bidireccional local (fase 1: motor headless)

```
Teams → auriculares → WASAPI loopback → [VAD A] → [ASR en] → [MT en→es] → consola/log/TCP
mic → [PTT F8 | VAD B] → [ASR es] → [MT es→en] → [Piper] → CABLE Input → Teams ("CABLE Output")
```

Sin PyTorch: faster-whisper/opus-mt = CTranslate2; Silero VAD, Parakeet, Piper = ONNX.

## Instalación (máquina nueva, Windows)

1. Python 3.11 + [uv](https://docs.astral.sh/uv/). Driver NVIDIA reciente + CUDA Toolkit 12.x.
2. cuDNN 9 (CUDA 12) y cudart ≥ 12.8: redist de NVIDIA en `~/tools/` (ver tabla de Entorno; rutas en `src/voztrad/cuda_dlls.py`).
3. [VB-Cable](https://vb-audio.com/Cable/) (pide admin). Después: `scripts\set_default_audio.ps1 -Match Speakers`.
4. ```powershell
   uv sync --extra ui
   .venv\Scripts\python.exe scripts\fetch_models.py   # VAD + voces Piper; ASR/MT se bajan solos
   .venv\Scripts\python.exe -m pytest tests -q
   ```

## Uso

**Día a día:** acceso directo "Traductor de voz" en el escritorio (= `pythonw -m voztrad app`).
Abre el motor sin consola + el overlay de subtítulos; cerrar el overlay (✕) apaga todo.
En Teams: micrófono = `CABLE Output`. Mantener **F8** para hablar (o pasar a manos libres desde el overlay).
Salida de consola del motor en `logs\engine-console.txt`, frases y latencias en `logs\run-*.jsonl`.

Overlay: arrastrar desde la barra superior (snap nativo de Windows), redimensionar desde la esquina,
A−/A+ tamaño de letra, clic derecho = opciones (original, latencia, opacidad, limpiar). Angosto → se apila.

**Latencia (medido, A = lo que dicen ellos):** subtítulo ~390 ms después de que terminan la
frase, castellano audible ~530 ms. Claves: Smart Turn v3.2 decide a los 200 ms de silencio si la
frase terminó (antes se esperaban 500 ms); MT en GPU; subtítulos en vivo mientras hablan.

**Traducción en dos niveles:** opus-mt (~30 ms) sale al instante; TranslateGemma-4B (llama.cpp,
~340 ms) reemplaza el subtítulo con una traducción mucho más fiel en modismos, y se usa para tu
voz cuando hay audio anterior sonando (no suma latencia). Parakeet va en fp16 para que entre todo
en 6 GB (total medido 4,46 GB). `--no-refine` lo apaga. Conviene cerrar juegos durante llamadas.

**Escuchar a los otros en castellano** (🔊 ESCUCHAR ES en el overlay, activado por defecto):
lo que dicen en inglés se dice en castellano por tus auriculares (~0,8 s después de cada frase,
medido) con Teams bajado al 35% mientras suena. La dirección A captura todo el audio del sistema
**menos el de voztrad** (process loopback EXCLUDE), así la traducción no se re-captura.
En manos libres/intérprete el mic se ignora mientras suena la traducción (por si usás parlantes).
`--no-listen` lo apaga; `--listen-voice models\es_AR-daniela-high.onnx` = voz rioplatense (más lenta).

**Tu voz clonada** (Kyutai Pocket TTS, ONNX en CPU, zero-shot): grabate una vez con
`python -m voztrad voice` (guía: 20 s en castellano + 20 s en inglés, chequea nivel, genera y
reproduce muestras). Queda en `models/voice/me.wav` (no se versiona). Sin grabación → Piper.
`--tts piper` fuerza la voz genérica (~190 ms menos de latencia, medido).
Uso permitido solo con tu propia voz / con consentimiento (licencia del modelo).

```powershell
.venv\Scripts\python.exe -m voztrad voice                         # grabar tu voz y elegir referencia
.venv\Scripts\python.exe -m voztrad app                           # motor + overlay
.venv\Scripts\python.exe -m voztrad ui                            # solo overlay (motor ya corriendo)
.venv\Scripts\python.exe -m voztrad run                           # motor headless; parakeet + PTT F8 por defecto
.venv\Scripts\python.exe -m voztrad run --asr whisper --b-mode vad # manos libres
.venv\Scripts\python.exe -m voztrad run --save-segments           # guarda cada frase en out/segments/
.venv\Scripts\python.exe -m voztrad bench out\segments\<run>      # whisper vs parakeet sobre el MISMO audio
.venv\Scripts\python.exe -m voztrad report logs\run-*.jsonl       # p50/p95 por etapa
.venv\Scripts\python.exe scripts\e2e.py --asr parakeet            # prueba E2E automática (A y B)
```

Flags útiles: `--hangover 500`, `--ptt-key F8`, `--mic "<substring>"`, `--no-a`, `--no-b`,
`--no-norm` (sin normalización rioplatense), `--no-gpu-warm`, `--parakeet-quant int8`.

Transporte para la UI (fase 2): JSON-lines en `127.0.0.1:8765`. Eventos `{"type": "phrase"|"status"|"drop", ...}`;
comandos `{"cmd": "ptt", "down": true}`, `{"cmd": "b_mode", "value": "vad"}`, `{"cmd": "stop"}`.

## Entorno (Windows) — lo que rompe y cómo quedó resuelto

| Problema | Solución |
|---|---|
| `onnxruntime-gpu` 1.30 de PyPI es **CUDA 13** → cae a CPU en silencio | 1.29.0 del feed CUDA 12 de Microsoft (`[tool.uv.index] ort-cuda12`) |
| `piper-tts` arrastra `onnxruntime` (CPU) que pisa al de GPU | `override-dependencies` en `pyproject.toml` |
| ORT CUDA 12 pide símbolos de cudart ≥ 12.8; toolkit instalado es 12.6 | cudart 12.9 redist en `~/tools/cuda_cudart-12.9.79`, precargado en `cuda_dlls.py` |
| cuDNN 9 no estaba | redist 9.27 en `~/tools/cudnn-9.27.0.42_cuda12/bin/x64` (lo usa ORT; CT2 trae su propio dispatcher) |
| Windows 11 hace power-throttling (EcoQoS) a procesos sin ventana: MT 45 → 190 ms tras inactividad | `winperf.disable_throttling()` al arrancar |
| ORT con spin-wait le roba CPU al MT (picos de 900 ms) | `allow_spinning=0` en las sesiones ORT |
| GPU cae a P8/210 MHz tras ~2 s ociosa: ASR 2-6× más lento | `gpuwarm.py`: carga la GPU solo mientras hay voz (0,6 s alcanzan). Alternativa admin: `nvidia-smi -lgc 1500,3105` |
| Instalar VB-Cable lo deja como salida por defecto (no oís nada) | `scripts\set_default_audio.ps1 -Match Speakers` |

Rutas de DLL configurables con `VOZTRAD_CUDART_BIN`, `VOZTRAD_CUDNN_BIN`, `VOZTRAD_CUDA_BIN`.
