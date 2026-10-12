#!/usr/bin/env bash
# voztrad-audio.sh - Create / remove the PipeWire audio routing for voice-bridge.
#
# Everything here is done with pactl on top of PipeWire and is fully reversible:
# `teardown` unloads exactly what `setup` loaded (tracked in a state file, with a
# name-based fallback). Nothing is written to any config; a reboot also clears it.
#
# Topology created by `setup`:
#
#   voztrad_calls (null sink)                 <- apps/call audio is routed here
#     +-- monitor --> loopback --> REAL_SINK  (you hear the call on your headphones)
#     +-- monitor ----------------> captured by the engine for translation
#         (audio sent straight to REAL_SINK, e.g. our own TTS, is NOT in this
#          monitor, so we never re-capture/translate our own output)
#
#   voztrad_mic (null sink)                   <- engine writes translated TTS here
#     +-- monitor --> remap-source "voztrad_mic_input" (apps see it as a microphone)
#
# Usage:
#   voztrad-audio.sh setup   [REAL_SINK]   # REAL_SINK defaults to the current default sink
#   voztrad-audio.sh teardown
#   voztrad-audio.sh status
#   voztrad-audio.sh record-calls OUT.wav [SECONDS]   # capture the calls monitor
#   voztrad-audio.sh app-volume "<app.name substring>" <PERCENT>  # e.g. "Teams" 40
set -euo pipefail

STATE_DIR="${XDG_RUNTIME_DIR:-/tmp}/voztrad"
STATE_FILE="$STATE_DIR/audio-modules.env"

CALLS_SINK="voztrad_calls"
MIC_SINK="voztrad_mic"
MIC_SOURCE="voztrad_mic_input"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

setup() {
  need pactl
  mkdir -p "$STATE_DIR"
  [ -f "$STATE_FILE" ] && die "already set up (state file $STATE_FILE exists); run teardown first"

  local real_sink="${1:-$(pactl get-default-sink)}"
  [ -n "$real_sink" ] || die "could not determine REAL_SINK"
  echo "Routing call audio to real sink: $real_sink"

  # 1) calls null sink + loopback of its monitor to the real output
  local m_calls m_loop
  m_calls=$(pactl load-module module-null-sink \
      sink_name="$CALLS_SINK" \
      sink_properties=device.description="$CALLS_SINK")
  m_loop=$(pactl load-module module-loopback \
      source="${CALLS_SINK}.monitor" sink="$real_sink" latency_msec=60)

  # 2) virtual mic: null sink + remap-source so apps enumerate it as a microphone
  local m_mic m_remap
  m_mic=$(pactl load-module module-null-sink \
      sink_name="$MIC_SINK" \
      sink_properties=device.description="$MIC_SINK")
  m_remap=$(pactl load-module module-remap-source \
      master="${MIC_SINK}.monitor" \
      source_name="$MIC_SOURCE" \
      source_properties=device.description='Voztrad Virtual Mic')

  {
    echo "M_CALLS=$m_calls"
    echo "M_LOOP=$m_loop"
    echo "M_MIC=$m_mic"
    echo "M_REMAP=$m_remap"
    echo "REAL_SINK=$real_sink"
  } > "$STATE_FILE"

  echo "Loaded modules: calls=$m_calls loop=$m_loop mic=$m_mic remap=$m_remap"
  echo "State saved to $STATE_FILE"
  status
}

teardown() {
  need pactl
  # Preferred: unload exactly the module ids we recorded.
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    . "$STATE_FILE"
    for m in "${M_LOOP:-}" "${M_CALLS:-}" "${M_REMAP:-}" "${M_MIC:-}"; do
      [ -n "$m" ] && pactl unload-module "$m" 2>/dev/null && echo "unloaded module $m" || true
    done
    rm -f "$STATE_FILE"
  else
    echo "no state file; falling back to name-based cleanup"
  fi
  # Fallback: unload anything still referencing our names (idempotent, safe to re-run).
  pactl list short modules \
    | grep -E "sink_name=(${CALLS_SINK}|${MIC_SINK})|source_name=${MIC_SOURCE}|source=${CALLS_SINK}\.monitor|master=${MIC_SINK}\.monitor" \
    | awk '{print $1}' \
    | while read -r id; do pactl unload-module "$id" 2>/dev/null && echo "unloaded leftover module $id" || true; done
  echo "teardown complete"
}

status() {
  echo "--- sinks ---";   pactl list short sinks   | grep -E "${CALLS_SINK}|${MIC_SINK}" || echo "(no voztrad sinks)"
  echo "--- sources ---"; pactl list short sources | grep -E "${CALLS_SINK}\.monitor|${MIC_SOURCE}|${MIC_SINK}\.monitor" || echo "(no voztrad sources)"
}

record_calls() {
  need pw-record
  local out="${1:?usage: record-calls OUT.wav [SECONDS]}" secs="${2:-5}"
  echo "Capturing ${CALLS_SINK} monitor for ${secs}s -> $out"
  # stream.capture.sink=true makes pw-record bind to the sink's MONITOR ports
  # (plain --target <name> binds to a source and would fall back to the mic).
  timeout "$secs" pw-record --target "$CALLS_SINK" -P '{ stream.capture.sink=true }' "$out"
}

app_volume() {
  need pactl
  local app="${1:?usage: app-volume \"<app.name substring>\" <PERCENT>}" pct="${2:?PERCENT required}"
  local idx
  idx=$(pactl list sink-inputs | awk -v app="$app" '
    /^Sink Input #/{idx=substr($3,2)}
    /application\.name = /{ if (index($0,app)) { print idx; exit } }')
  [ -n "$idx" ] || die "no sink-input whose application.name contains '$app'"
  echo "sink-input #$idx ($app) -> ${pct}%"
  pactl set-sink-input-volume "$idx" "${pct}%"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  setup)        setup "$@" ;;
  teardown)     teardown ;;
  status)       status ;;
  record-calls) record_calls "$@" ;;
  app-volume)   app_volume "$@" ;;
  *) echo "usage: $0 {setup [REAL_SINK]|teardown|status|record-calls OUT.wav [SECONDS]|app-volume \"<app>\" <PERCENT>}" >&2; exit 2 ;;
esac
