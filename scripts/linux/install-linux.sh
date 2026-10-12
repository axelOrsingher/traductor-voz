#!/usr/bin/env bash
# One-shot Linux install for voice-bridge:
#   1. create the project venv (uv sync) with the CUDA 12 stack
#   2. fetch the models the MVP needs (Silero VAD + a Piper English voice)
#   3. add the user to the 'input' group (global F8 hotkey via evdev)
#   4. install a desktop launcher ("voice-bridge") in the app menu + on the Desktop
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"

command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

echo "==> creating venv (uv sync --extra ui)"
uv sync --extra ui

echo "==> fetching models (Silero VAD + Piper en_US-lessac-medium)"
.venv/bin/python - <<'PY'
import shutil, pathlib, site, sys
sys.path.insert(0, "src")
models = pathlib.Path("models"); models.mkdir(exist_ok=True)
# Silero VAD onnx ships inside the silero-vad wheel
import silero_vad
sv = pathlib.Path(silero_vad.__file__).parent / "data" / "silero_vad.onnx"
shutil.copy(sv, models / "silero_vad.onnx")
from huggingface_hub import hf_hub_download
base = "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
for f in (base, base + ".json"):
    p = hf_hub_download("rhasspy/piper-voices", f)
    shutil.copy(p, models / pathlib.Path(f).name)
print("models ready:", sorted(p.name for p in models.iterdir()))
PY

echo "==> adding $USER to the 'input' group (global F8 hotkey)"
if ! id -nG "$USER" | grep -qw input; then
  sudo usermod -aG input "$USER"
  echo "   added — log out/in once for it to take effect"
fi

echo "==> installing desktop launcher"
APPS="$HOME/.local/share/applications"; mkdir -p "$APPS"
DESK="$APPS/voice-bridge.desktop"
cat > "$DESK" <<EOF
[Desktop Entry]
Type=Application
Name=voice-bridge
Comment=Real-time two-way voice translation for calls
Exec=$HERE/voice-bridge
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
EOF
chmod +x "$DESK" "$HERE/voice-bridge"
# also drop a copy on the Desktop if there is one
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ]; then
  cp "$DESK" "$DESKTOP_DIR/voice-bridge.desktop"
  chmod +x "$DESKTOP_DIR/voice-bridge.desktop"
  gio set "$DESKTOP_DIR/voice-bridge.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "done. Launch 'voice-bridge' from the app menu or the desktop icon."
echo "In your call app, set the microphone to: voztrad_mic_input"
