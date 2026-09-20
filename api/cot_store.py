"""On-disk chain-of-thought side-store for webui sessions.

Long inference sessions persist each assistant message's reasoning in the
main session sidecar. After the #4765 CoT dedup the text is stored once,
but a 261k-message session still carries ~660 MB of reasoning text that
the UI only shows when a thinking card is expanded. This module moves that
text out of the main sidecar into a sibling file pair::

    <sid>.cot     -- concatenated UTF-8 reasoning records (one per message index)
    <sid>.cot.idx -- JSON {"v":1, "count":N, "offsets":[...]} (dense: every
                     message index has an offset; empty record == no CoT)

The dense index keeps ``record index == message position`` so callers
never need a second mapping, and selective reads are two ``seek``/``read``
calls per index — O(k), independent of session size.

Invariants:
  * The main sidecar's message list stays the source of truth for ORDER and
    COUNT; the side-store is a content side-channel keyed by position.
  * Writers are ``write_full`` (rebuild) and ``append_records`` (tail grow).
    ``sync()`` picks the cheap safe path.
  * Readers validate the index against the record file size and fail closed
    (missing/corrupt store == no CoT available, never a broken session).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_COT_VERSION = 1
_IDX_SUFFIX = ".cot.idx"
_COT_SUFFIX = ".cot"


def _paths(session_dir: Path, session_id: str) -> tuple[Path, Path]:
    return (
        Path(session_dir) / f"{session_id}{_COT_SUFFIX}",
        Path(session_dir) / f"{session_id}{_IDX_SUFFIX}",
    )


def _read_index(index_path: Path) -> dict | None:
    """Load and validate the offset index, or None when absent/invalid."""
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("v") != _COT_VERSION:
        return None
    count = raw.get("count")
    offsets = raw.get("offsets")
    if not isinstance(count, int) or count < 0:
        return None
    if not isinstance(offsets, list) or len(offsets) != count:
        return None
    if any(not isinstance(o, int) or o < 0 for o in offsets):
        return None
    return {"count": count, "offsets": offsets}


def load_index(session_dir: Path, session_id: str) -> dict | None:
    """Return the validated side-store index for a session, or None."""
    _, index_path = _paths(session_dir, session_id)
    if not index_path.exists():
        return None
    return _read_index(index_path)


def record_count(session_dir: Path, session_id: str) -> int:
    """Number of message positions covered by the side store (0 if absent)."""
    idx = load_index(session_dir, session_id)
    return idx["count"] if idx else 0


def read_records(
    session_dir: Path, session_id: str, indices: list[int]
) -> dict[int, str]:
    """Read specific CoT records by message index.

    Returns ``{index: text}`` for every index in range; out-of-range or
    unreadable positions are simply absent from the result.
    """
    idx = load_index(session_dir, session_id)
    if idx is None:
        return {}
    cot_path, _ = _paths(session_dir, session_id)
    try:
        size = cot_path.stat().st_size
    except OSError:
        return {}
    offsets = idx["offsets"]
    count = idx["count"]
    out: dict[int, str] = {}
    try:
        with open(cot_path, "rb") as fh:
            for i in indices:
                if not isinstance(i, int) or i < 0 or i >= count:
                    continue
                # offsets[k] = byte length AFTER message k, so record i is
                # [offsets[i-1] (or 0), offsets[i]).
                start = 0 if i == 0 else offsets[i - 1]
                end = offsets[i]
                if start > size or end < start:
                    continue
                fh.seek(start)
                data = fh.read(end - start) if end > start else b""
                text = data.decode("utf-8", errors="replace")
                if text:
                    out[i] = text
    except OSError:
        return out
    return out


def _write_atomic(path: Path, data: bytes) -> None:
    """Write bytes to a temp file in the same directory, then atomically replace."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".cot-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _has_inline_reasoning(message: dict) -> bool:
    r = message.get("reasoning")
    return isinstance(r, str) and bool(r.strip())


def _inline_reasoning_indices(messages: list) -> set[int]:
    return {
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and _has_inline_reasoning(m)
    }


def write_full(session_dir: Path, session_id: str, messages: list) -> int:
    """Rebuild the side store from ``messages`` and return the record count.

    A message's CoT is taken from its inline ``reasoning`` field when
    present; otherwise, when a previous store covers the position and the
    message is marked ``_has_cot``, the record is carried over from the old
    store (this keeps rebuilds lossless after lazy-stripped loads).
    """
    old = load_index(session_dir, session_id)
    cot_path, index_path = _paths(session_dir, session_id)
    old_fh = None
    if old is not None:
        try:
            old_fh = open(cot_path, "rb")
        except OSError:
            old = None
            old_fh = None

    offsets: list[int] = []
    buf = bytearray()
    try:
        for m in messages:
            text = ""
            if isinstance(m, dict):
                r = m.get("reasoning")
                if isinstance(r, str) and r.strip():
                    text = r
                elif m.get("_has_cot") and old is not None and old_fh is not None:
                    i = len(offsets)
                    if i < old["count"]:
                        # record i is [offsets[i-1] (or 0), offsets[i]).
                        s = 0 if i == 0 else old["offsets"][i - 1]
                        e = old["offsets"][i]
                        try:
                            old_fh.seek(s)
                            text = old_fh.read(e - s).decode("utf-8", "replace")
                        except OSError:
                            text = ""
            buf.extend(text.encode("utf-8"))
            offsets.append(len(buf))
    finally:
        if old_fh is not None:
            old_fh.close()

    _write_atomic(cot_path, bytes(buf))
    _write_atomic(
        index_path,
        json.dumps({"v": _COT_VERSION, "count": len(offsets), "offsets": offsets},
                   separators=(",", ":")).encode("utf-8"),
    )
    return len(offsets)


def append_records(session_dir: Path, session_id: str, new_messages: list) -> int:
    """Append CoT records for freshly appended messages (tail-grow fast path).

    The caller guarantees ``new_messages`` are the messages at positions
    ``old_count .. old_count+len(new_messages)-1`` (an append, not a rewrite).
    Returns the new total count.
    """
    idx = load_index(session_dir, session_id)
    if idx is None:
        return write_full(session_dir, session_id, new_messages)
    cot_path, index_path = _paths(session_dir, session_id)
    try:
        size = cot_path.stat().st_size
    except OSError:
        return write_full(session_dir, session_id, new_messages)
    if idx["offsets"][-1] > size and idx["count"]:
        # Truncated record file — rebuild to stay consistent.
        return write_full(session_dir, session_id, new_messages)

    buf = bytearray()
    offsets = list(idx["offsets"])
    for m in new_messages:
        text = ""
        if isinstance(m, dict):
            r = m.get("reasoning")
            if isinstance(r, str) and r.strip():
                text = r
            elif m.get("_has_cot"):
                # Carried over position (no inline text): keep the old record.
                i = len(offsets) - 1
                if 0 <= i < idx["count"]:
                    s = 0 if i == 0 else idx["offsets"][i - 1]
                    e = idx["offsets"][i]
                    if s <= size:
                        try:
                            with open(cot_path, "rb") as fh:
                                fh.seek(s)
                                text = fh.read(max(0, e - s)).decode("utf-8", "replace")
                        except OSError:
                            text = ""
        buf.extend(text.encode("utf-8"))
        offsets.append(size + len(buf))

    with open(cot_path, "ab") as fh:
        fh.write(bytes(buf))
        # Authoritative post-append size: the last offset MUST equal the real
        # file size, or the freshly appended record reads as empty (its end
        # would be computed from the pre-append size captured above).
        new_size = fh.tell()
    offsets[-1] = new_size
    _write_atomic(
        index_path,
        json.dumps(
            {"v": _COT_VERSION, "count": len(offsets), "offsets": offsets},
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    return len(offsets)


def sync(session_dir: Path, session_id: str, messages: list) -> str:
    """Bring the side store in line with ``messages``. Returns the path taken.

    Paths:
      * ``"noop"``     -- store already matches (count equal, no inline CoT);
      * ``"append"``   -- tail grow: the new messages' inline CoT is appended;
      * ``"full"``     -- rebuild (new session, compaction shrink, mid-history
                          edit, or a stripped store being rehydrated);
      * ``"removed"``  -- nothing to store; a stale store was deleted.
    """
    n = len(messages)
    inline = _inline_reasoning_indices(messages)
    idx = load_index(session_dir, session_id)

    if idx is not None and idx["count"] == n and not inline:
        return "noop"

    if idx is not None and n > idx["count"]:
        append_records(session_dir, session_id, messages[idx["count"]:])
        return "append"

    if idx is None and not inline:
        return "removed"

    write_full(session_dir, session_id, messages)
    return "full"


def remove(session_dir: Path, session_id: str) -> None:
    """Delete both side-store files (best effort)."""
    for p in _paths(session_dir, session_id):
        try:
            if p.exists():
                os.unlink(p)
        except OSError:
            pass
