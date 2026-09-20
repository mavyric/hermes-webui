"""Regression tests for the size-aware SESSIONS LRU cap and the cheap
save() prefix count read (#4765 follow-up).

Two production defects fixed here (mavnlbox 2026-09-20 incident):

1. The SESSIONS LRU was capped by ENTRY COUNT only (300). A single very long
   session (measured: 261,158 messages, 1.9 GiB sidecar, ~7.4 GiB parsed RSS)
   pinned several GiB resident while idle/active and the count cap never
   bounded it. A total-bytes cap (``webui.sessions_cache_size_bytes``, default
   3 GiB) now evicts the largest idle sessions by size via the same proven
   evictability gate (never an active/unsaved entry).

2. ``Session.save()`` read the ENTIRE sidecar and ``json.loads``-ed it on every
   save just to recover the existing message count for the #1558 backup guard.
   For a multi-GiB sidecar that is a full parse (peak memory + CPU) on every
   save. The count now comes from the cheap metadata prefix (``message_count``
   is written BEFORE the messages array), and the full file is read only when
   the count is unknown (legacy layout) or a shrink requires the .bak bytes.

These tests prove:
  * the byte cap is configurable via webui.sessions_cache_size_bytes and falls
    back to the bounded default on bad values;
  * eviction fires when total estimated bytes exceed the cap EVEN WHEN the
    entry count is under the count cap (the exact incident geometry: one huge
    session + few small ones);
  * an active/streaming session is NEVER evicted by the size cap (safety
    invariant inherited from #4765);
  * an evicted-by-size session lazily reloads from disk intact;
  * save() of a large session does NOT read the full file in the common
    grow/flat case (0 full reads), and STILL writes the .bak on a shrink
    (legacy and modern layouts), preserving the #1558 guarantee.
"""
import collections
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_session_env():
    """Isolate all SESSIONS-cache global state onto a throwaway temp dir."""
    from api import config as _cfg
    from api import models as _models

    tmpdir = tempfile.mkdtemp()
    sessions_dir = Path(tmpdir) / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    old = {
        "cfg_SESSION_DIR": _cfg.SESSION_DIR,
        "models_SESSION_DIR": getattr(_models, "SESSION_DIR", None),
        "cfg_SESSION_INDEX_FILE": _cfg.SESSION_INDEX_FILE,
        "models_SESSION_INDEX_FILE": getattr(_models, "SESSION_INDEX_FILE", None),
        "SESSIONS": _cfg.SESSIONS,
        "LOCK": _cfg.LOCK,
        "SESSIONS_MAX": _cfg.SESSIONS_MAX,
        "cfg": getattr(_cfg, "cfg", None),
    }

    index_file = sessions_dir / "_index.json"
    _cfg.SESSION_DIR = sessions_dir
    _models.SESSION_DIR = sessions_dir
    _cfg.SESSION_INDEX_FILE = index_file
    _models.SESSION_INDEX_FILE = index_file
    _cfg.LOCK = threading.Lock()
    _models.LOCK = _cfg.LOCK
    _cfg.SESSIONS = collections.OrderedDict()
    _models.SESSIONS = _cfg.SESSIONS

    try:
        yield sessions_dir
    finally:
        _cfg.SESSION_DIR = old["cfg_SESSION_DIR"]
        if old["models_SESSION_DIR"] is not None:
            _models.SESSION_DIR = old["models_SESSION_DIR"]
        _cfg.SESSION_INDEX_FILE = old["cfg_SESSION_INDEX_FILE"]
        if old["models_SESSION_INDEX_FILE"] is not None:
            _models.SESSION_INDEX_FILE = old["models_SESSION_INDEX_FILE"]
        _cfg.SESSIONS = old["SESSIONS"]
        _models.SESSIONS = old["SESSIONS"]
        _cfg.LOCK = old["LOCK"]
        _models.LOCK = old["LOCK"]
        _cfg.SESSIONS_MAX = old["SESSIONS_MAX"]
        if old["cfg"] is not None:
            _cfg.cfg = old["cfg"]
        shutil.rmtree(tmpdir, ignore_errors=True)


def _make_persisted_session(idx, *, n_messages=2):
    """Build + save a real session with real messages (so it persists)."""
    from api.models import Session

    messages = [
        {"role": "user", "content": f"hello {idx}-{j}", "timestamp": time.time()}
        for j in range(max(1, n_messages))
    ]
    s = Session(session_id=f"sess{idx:04d}", title=f"Session {idx}", messages=messages)
    s.save()
    return s


def _insert(sid_session):
    """Insert a session into the cache exactly like the production accessors do."""
    from api.config import SESSIONS, LOCK
    from api.models import _evict_sessions_over_cap

    with LOCK:
        SESSIONS[sid_session.session_id] = sid_session
        SESSIONS.move_to_end(sid_session.session_id)
        _evict_sessions_over_cap()


def _make_large_sidecar(path: Path, n_messages: int) -> Path:
    """Write a session sidecar whose messages array dominates its size."""
    payload = {
        "session_id": path.stem.replace(".json", ""),
        "title": "large",
        "message_count": n_messages,
        "messages": [
            {"role": "assistant", "content": "x" * 2000, "timestamp": time.time()}
            for _ in range(n_messages)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ─────────────────────────── config knob ────────────────────────────────────

def test_size_cap_reads_config_yaml_key():
    """The byte cap is configurable via webui.sessions_cache_size_bytes."""
    from api import config as _cfg

    assert _cfg.get_sessions_cache_size_bytes({"webui": {"sessions_cache_size_bytes": 123456}}) == 123456
    # Invalid / missing values must fall back to the bounded default, never
    # disable the bound.
    default = _cfg.DEFAULT_SESSIONS_CACHE_SIZE_BYTES
    assert isinstance(default, int) and default >= 1
    assert _cfg.get_sessions_cache_size_bytes({"webui": {}}) == default
    assert _cfg.get_sessions_cache_size_bytes({}) == default
    for bad in ("nope", 0, -5, None, ""):
        assert _cfg.get_sessions_cache_size_bytes({"webui": {"sessions_cache_size_bytes": bad}}) == default


# ─────────────────────── invariant 1: size-based eviction ───────────────────

def test_eviction_fires_on_total_bytes_below_count_cap(isolated_session_env):
    """ONE huge session + few small ones must be bounded by BYTES, not count.

    This is the incident geometry: 1–3 entries (well under the 300-entry cap)
    holding several GiB. After the fix the idle large session is evicted once
    the estimated resident bytes exceed the cap.
    """
    from api import config as _cfg
    from api.config import SESSIONS
    from api.models import Session, _session_estimated_bytes

    _cfg.SESSIONS_MAX = 300  # generous count cap — count alone would never fire
    # Tiny byte cap so the big session's estimate (sidecar x 4) exceeds it.
    _cfg.get_sessions_cache_size_bytes = lambda config_data=None: 500_000

    big = _make_persisted_session(900, n_messages=2000)  # ~216 KB sidecar -> ~866 KB est
    assert _session_estimated_bytes(big) > 500_000
    small = _make_persisted_session(901, n_messages=2)
    _insert(big)
    _insert(small)

    assert len(SESSIONS) <= 2
    # The large idle session must have been evicted by the byte cap.
    assert big.session_id not in SESSIONS, (
        "a multi-MB idle session survived a 1 MB byte cap — the count-only "
        "cap bug (incident 2026-09-20) is not fixed"
    )
    # The small session (newest, under budget) is retained.
    assert small.session_id in SESSIONS

    # Evicted session lazily reloads from disk intact.
    reloaded = Session.load(big.session_id)
    assert reloaded is not None
    assert len(reloaded.messages) == 2000


def test_active_session_survives_size_cap(isolated_session_env):
    """An actively streaming session is NEVER evicted by the byte cap."""
    from api import config as _cfg
    from api.config import SESSIONS

    _cfg.SESSIONS_MAX = 300
    _cfg.get_sessions_cache_size_bytes = lambda config_data=None: 500_000

    active = _make_persisted_session(910, n_messages=2000)
    active.active_stream_id = "live-stream-xyz"
    active.pending_user_message = "in-flight question"
    active.pending_started_at = time.time()
    _insert(active)
    # Flood with more large idle sessions — they must yield, not the live one.
    for i in range(911, 920):
        _insert(_make_persisted_session(i, n_messages=2000))

    assert active.session_id in SESSIONS, (
        "an actively streaming session was evicted by the size cap — this "
        "would drop an in-flight turn (#4765 safety invariant)"
    )
    assert SESSIONS[active.session_id] is active


def test_size_cap_does_not_evict_when_under_budget(isolated_session_env):
    """Small sessions under the byte budget survive eviction passes."""
    from api import config as _cfg
    from api.config import SESSIONS

    _cfg.SESSIONS_MAX = 300
    _cfg.get_sessions_cache_size_bytes = lambda config_data=None: 100_000_000  # 100 MB

    created = [_make_persisted_session(i, n_messages=2) for i in range(5)]
    for s in created:
        _insert(s)
    assert len(SESSIONS) == 5, "sessions under the byte budget were wrongly evicted"


# ─────────────────────── invariant 2: save() cheap count ────────────────────

def _count_full_reads(session_path, monkeypatch):
    """Record every Path.read_text() call (any instance) via the class.

    ``save()`` recovers the existing message count via ``self.path.read_text``;
    the cheap-prefix path instead uses ``open()``, so read_text() calls are the
    observable signal of a full-file read. Returns the shared call list.
    """
    from pathlib import Path

    calls = []
    real_read_text = Path.read_text

    def spy(self, *a, **k):
        calls.append(self)
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy, raising=True)
    return calls


def test_save_large_session_does_not_full_read_on_grow(isolated_session_env, monkeypatch):
    """Growing/flat saves of a large session must NOT read the full file.

    The pre-fix code read + json.loads-ed the entire multi-GiB sidecar on every
    save. Observable behavior after the fix: zero read_text() calls on the
    sidecar for a grow/flat save when the prefix carries message_count.
    """
    from api.models import Session

    calls = _count_full_reads(None, monkeypatch)

    n = 300
    s = Session(
        session_id="biggrow001",
        title="big",
        messages=[{"role": "user", "content": "x" * 1000, "timestamp": time.time()} for _ in range(n)],
    )
    s.save()  # first save: file did not exist -> no read expected

    before = s.path.stat().st_size
    assert before > 100_000, "sidecar must be large enough to make the full read measurable"
    calls.clear()

    # Grow the conversation and save again (common case: count goes UP).
    s.messages.append({"role": "assistant", "content": "y" * 1000, "timestamp": time.time()})
    s.save()

    assert calls == [], (
        f"save() read the full {before // 1024} KB sidecar to recover the "
        f"message count — the cheap-prefix path was not used: {calls}"
    )
    # The write still landed correctly.
    reloaded = Session.load(s.session_id)
    assert len(reloaded.messages) == n + 1


def test_save_shrink_still_writes_bak(isolated_session_env, monkeypatch):
    """A shrink (incoming < existing) must still write the #1558 .bak."""
    from api.models import Session

    _count_full_reads(None, monkeypatch)

    n = 50
    s = Session(
        session_id="shrink0001",
        title="shrink",
        messages=[{"role": "user", "content": f"m{i}", "timestamp": time.time()} for i in range(n)],
    )
    s.save()
    bak = s.path.with_suffix(".json.bak")
    assert not bak.exists()

    # Simulate a shrink: drop most messages in memory and save.
    s.messages = s.messages[:3]
    s.save()
    assert bak.exists(), "shrink did not write the #1558 recovery backup"
    bak_data = json.loads(bak.read_text(encoding="utf-8"))
    assert len(bak_data.get("messages") or []) == n, "backup lost the pre-shrink history"


def test_save_shrink_writes_bak_for_legacy_layout(isolated_session_env, monkeypatch):
    """Legacy sidecars (no prefix message_count) fall back to full read + bak."""
    from api.models import Session

    calls = _count_full_reads(None, monkeypatch)

    n = 40
    # Write a legacy-layout sidecar: no message_count key at all.
    legacy = {
        "session_id": "legacy0001",
        "title": "legacy",
        "messages": [{"role": "user", "content": f"m{i}", "timestamp": time.time()} for i in range(n)],
    }
    (isolated_session_env / "legacy0001.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = Session.load("legacy0001")
    assert loaded is not None and len(loaded.messages) == n
    calls.clear()

    loaded.messages = loaded.messages[:2]
    loaded.save()
    # Legacy prefix has no count -> full read is REQUIRED to know the count.
    assert len(calls) >= 1, "legacy layout must fall back to the full-file read"
    bak = (isolated_session_env / "legacy0001.json.bak")
    assert bak.exists(), "legacy shrink did not write the #1558 recovery backup"
