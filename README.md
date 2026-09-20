# Hermes Web UI — mavyric fork

This fork (`mavyric/hermes-webui`) tracks
[nesquena/hermes-webui](https://github.com/nesquena/hermes-webui) and is
re-synced to it daily by the `sync-with-upstream` GitHub Actions workflow.

The complete product documentation — features, setup, configuration, Docker,
Nix, troubleshooting — lives in the
[upstream README](https://github.com/nesquena/hermes-webui#readme).

This file records only what this fork adds on top of upstream.

---

## Large-session memory containment (2026-09, upstream #4765)

Very large sessions (hundreds of thousands of messages) previously pinned the
entire WebUI process in RAM and thrashed the host. This fork adds:

- **Byte-aware session cache** — LRU eviction is aware of message count *and*
  estimated bytes (`webui.sessions_cache_size_bytes`, default 3 GiB);
  active/unsaved sessions are protected.
- **Cheap `save()`** — sidecar message counts are read from the metadata
  prefix instead of re-parsing the full multi-GB JSON on every save.
- **CoT dedup at rest** — reasoning is persisted once (`reasoning` is
  canonical; identical `reasoning_content` and WebUI-unused `api_content`
  are dropped; provider-facing replay reconstructs `reasoning_content`).
  Tested on a real 261k-message session: sidecar **1.899 GB → 0.737 GB**
  with all messages and 260k+ CoT blocks preserved.

## Lazy CoT side store (2026-09, #4765 follow-up)

Chain-of-thought text no longer rides in the session sidecar at all:

- Reasoning moves to sibling `<sid>.cot` / `<sid>.cot.idx` files; the
  sidecar keeps only visible text plus a `_has_cot` marker.
- The UI renders a collapsed Thinking card and fetches only the records it
  needs via `GET /api/session/reasoning` on expand — opening a large
  conversation no longer materializes its CoT in RAM.
  Tested on a real 261k-message session: full-load peak **~7.4 GiB →
  ~0.3 GiB**; sidecar **→ ~72 MB**.
- Provider replay rehydrates CoT selectively, only for messages it sends.
- Duplicating a session carries the CoT store over (`/api/session/duplicate`).
- `_has_cot` rows are renderable anchors, so windows of thinking-only turns
  open on real (collapsed) Thinking cards instead of the empty state.

## Upstream sync + fork-based updates

- **Daily `sync-with-upstream` GitHub Actions workflow** — fast-forwards or
  rebases fork-local commits onto `upstream/master`, pushes with
  `--force-with-lease`; on conflict the fork's `master` is left untouched for
  manual resolution.
- **Fork-based update flow** — WebUI update checks source releases from this
  fork (which mirrors upstream) and strip embedded credentials from any
  remote URL they expose.
