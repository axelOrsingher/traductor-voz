# CLAUDE.md — rules for any agent working on this repo

voztrad (voice-bridge): local, real-time bidirectional en<->es voice translator for calls
(Teams etc.). Several agents work on it from different machines (Windows, Linux, macOS);
these rules keep that work consistent.

## Language
- Everything in the repo is **English**: identifiers, comments, docstrings, log/status
  messages, CLI help, docs, commit messages, PR descriptions. (The owner chats in Spanish;
  answer them in Spanish.) Exception: Spanish *content* the product handles on purpose
  (voseo.py rules, translation prompts, Spanish test phrases).
- User-facing UI text and the voice-recording guide will be localizable (en/es, chosen
  independently). Don't hardcode new user-facing strings in only one language.

## Engineering rules
- **No PyTorch**, anywhere. Inference is CTranslate2, ONNX Runtime, llama.cpp (and, on
  Apple Silicon, MLX / Metal is acceptable).
- **Measure, don't assume.** Every latency/VRAM/quality claim in code comments, docs or
  commits comes from a run on real hardware; say which machine. Report p50/p95.
- Latency is the product. A change must not make another platform slower; if you touch
  shared code, say so in the PR and how you checked the other platforms didn't regress.
- Platform code lives in `*_win.py` / `*_linux.py` / `*_mac.py` behind a neutral module
  (`audio.py`, `ptt.py`, `ducker.py`, ...). The engine never imports OS-specific modules.
  Missing capabilities degrade with a clear status message instead of crashing.
- Hardware-dependent defaults go through `hwdetect.choose_profile()`; explicit CLI flags
  always win, and `--no-auto` falls back to `Config` defaults.
- Keep the callback threads of audio backends trivial (enqueue only): a slow callback
  glitches the real call audio.

## Workflow
- Work on a branch, open a PR to `main`. Small focused commits.
- Before a PR: `python -m pytest tests -q` passes, and an end-to-end latency run on your
  machine is in the PR description (before/after when you change the hot path).
- Porting notes and measurements: `docs/porting/<platform>-*.md`.
