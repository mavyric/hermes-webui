"""#4765 follow-up: on-disk CoT side store (lazy reasoning).

Regression coverage for:
  * api/cot_store.py — index/record round-trip, selective read, sync
    fast paths, carry-over on rebuild.
  * Session.load()  — legacy sidecar migration (extract -> strip -> persist,
    ``_has_cot`` marker) and idempotent re-load.
  * Session.save()  — store stays in lockstep (tail append, no-op steady
    state).
  * _sanitize_messages_for_api — CoT rehydration for model replay.
  * GET /api/session/reasoning — lazy fetch endpoint (no session load).
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api import cot_store  # noqa: E402


def _make_sidecar(directory: Path, sid: str, messages, **meta) -> Path:
    payload = {
        "session_id": sid,
        "messages": messages,
        "tool_calls": [],
        "updated_at": 1.0,
        **meta,
    }
    p = directory / f"{sid}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def _msg(role, content="", reasoning=None, **extra):
    m = {"role": role, "content": content}
    if reasoning is not None:
        m["reasoning"] = reasoning
    m.update(extra)
    return m


# ── cot_store primitives ────────────────────────────────────────────────────

def test_cot_roundtrip_selective_read():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        sid = "s-round"
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "a1", "reasoning-one"),
            _msg("assistant", "a2", "reasoning-two"),
            _msg("user", "again"),
            _msg("assistant", "a3"),  # no reasoning -> empty record
        ]
        n = cot_store.write_full(d, sid, msgs)
        assert n == 5
        got = cot_store.read_records(d, sid, [1, 2, 4])
        # index 4 has no reasoning -> empty record -> omitted from the result
        assert got == {1: "reasoning-one", 2: "reasoning-two"}
        # utf-8 multibyte safety
        msgs2 = [_msg("assistant", "", "héllo ✓ \n line2")]
        cot_store.write_full(d, "s-utf", msgs2)
        assert cot_store.read_records(d, "s-utf", [0]) == {0: "héllo ✓ \n line2"}


def test_cot_sync_fast_paths():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        sid = "s-sync"
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "a", "r1"),
        ]
        assert cot_store.sync(d, sid, msgs) == "full"
        # steady state: stripped list, same count -> noop
        stripped = [_msg("user", "hi"), _msg("assistant", "a", _has_cot=True)]
        assert cot_store.sync(d, sid, stripped) == "noop"
        # tail grow -> append
        grown = stripped + [_msg("assistant", "b", "r2")]
        assert cot_store.sync(d, sid, grown) == "append"
        # grown is 3 messages: idx0 user, idx1 assistant(r1), idx2 assistant(r2)
        assert cot_store.read_records(d, sid, [1, 2]) == {1: "r1", 2: "r2"}
        # fresh session (no store) with no inline cot -> removed
        assert cot_store.sync(d, "s-none", [_msg("user", "x")]) == "removed"


def test_cot_write_full_carry_over_stripped():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        sid = "s-carry"
        cot_store.write_full(d, sid, [_msg("assistant", "a", "kept-one"),
                                      _msg("assistant", "b", "kept-two")])
        # rebuild from a lazy-stripped list: markers carry records forward
        stripped = [_msg("assistant", "a", _has_cot=True),
                    _msg("assistant", "b", _has_cot=True),
                    _msg("assistant", "c", "fresh-three")]
        n = cot_store.write_full(d, sid, stripped)
        assert n == 3
        assert cot_store.read_records(d, sid, [0, 1, 2]) == {
            0: "kept-one", 1: "kept-two", 2: "fresh-three"}


def test_cot_large_scale_selective_read():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        n = 300_000
        body = "x" * 500
        msgs = [_msg("assistant", "a", body)] * n
        cot_store.write_full(d, "s-big", msgs)
        t0 = time.monotonic()
        got = cot_store.read_records(d, "s-big", list(range(n - 30, n)))
        dt = time.monotonic() - t0
        assert len(got) == 30
        assert dt < 2.0, f"selective read of 30/300k took {dt:.2f}s"


# ── Session.load migration ─────────────────────────────────────────────────

def test_session_load_migrates_legacy_sidecar(tmp_path, monkeypatch):
    from api import models
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    sid = "s-migrate"
    msgs = [
        _msg("user", "hi"),
        _msg("assistant", "hello", "deep reasoning A"),
        _msg("assistant", "", "deep reasoning B"),
    ]
    _make_sidecar(tmp_path, sid, msgs)

    s = models.Session.load(sid)
    assert s is not None
    # sidecar rewritten without inline reasoning, markers present
    on_disk = json.loads((tmp_path / f"{sid}.json").read_text())["messages"]
    assert all("reasoning" not in m for m in on_disk)
    assert on_disk[1]["_has_cot"] and on_disk[2]["_has_cot"]
    # in-memory session slimmed too
    assert "reasoning" not in s.messages[1]
    assert s.messages[1]["_has_cot"] is True
    # side store populated
    assert cot_store.read_records(tmp_path, sid, [1, 2]) == {
        1: "deep reasoning A", 2: "deep reasoning B"}
    # idempotent re-load: no inline reasoning on disk -> no rewrite churn
    before = (tmp_path / f"{sid}.json").stat().st_size
    s2 = models.Session.load(sid)
    assert s2.messages[1]["_has_cot"] is True
    after = (tmp_path / f"{sid}.json").stat().st_size
    assert before == after
    # store intact
    assert cot_store.read_records(tmp_path, sid, [1]) == {1: "deep reasoning A"}


# ── Session.save store lockstep ────────────────────────────────────────────

def test_session_save_keeps_store_in_lockstep(tmp_path, monkeypatch):
    from api import models
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    sid = "s-save"
    msgs = [_msg("user", "hi"), _msg("assistant", "a", "r1")]
    _make_sidecar(tmp_path, sid, msgs)
    s = models.Session.load(sid)  # migrates + strips
    # simulate a tail append (new assistant turn with inline reasoning)
    s.messages.append(_msg("assistant", "b", "r2"))
    s.save(touch_updated_at=False, skip_index=True)
    recs = cot_store.read_records(tmp_path, sid, [1, 2])
    assert recs == {1: "r1", 2: "r2"}
    # save() must call the store sync (spy proves the wiring)
    calls = []
    real_sync = cot_store.sync

    def spy(*a, **k):
        calls.append(a)
        return real_sync(*a, **k)

    with patch.object(cot_store, "sync", side_effect=spy):
        s.save(touch_updated_at=False, skip_index=True)
    assert calls, "save() must call cot_store.sync"


# ── replay rehydration ──────────────────────────────────────────────────────

def test_sanitize_rehydrates_lazy_cot(tmp_path, monkeypatch):
    from api import models, streaming
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    sid = "s-replay"
    msgs = [
        _msg("user", "hi"),
        _msg("assistant", "a", "reasoning for replay"),
    ]
    _make_sidecar(tmp_path, sid, msgs)
    s = models.Session.load(sid)  # strips reasoning -> _has_cot
    out = streaming._sanitize_messages_for_api(s.messages, session_id=sid)
    # provider sees reasoning_content reconstructed from the side store
    asst = [m for m in out if m.get("role") == "assistant"]
    assert asst and asst[0].get("reasoning_content") == "reasoning for replay"
    assert asst[0].get("_has_cot") is None, "internal marker must not leak to the API"
    # without a session_id the projection falls back to no-CoT (no crash)
    out2 = streaming._sanitize_messages_for_api(s.messages)
    asst2 = [m for m in out2 if m.get("role") == "assistant"]
    assert asst2 and "reasoning_content" not in asst2[0]


# ── /api/session/reasoning endpoint ─────────────────────────────────────────

def _make_handler():
    captured = {}

    class _Wfile:
        def __init__(self, sink):
            self._sink = sink

        def write(self, data):
            self._sink["body"] = self._sink.get("body", b"") + data

    class H:
        headers = {}
        _trusted_auth_session_cookie_value = None

        def __init__(self):
            self.wfile = _Wfile(captured)

        def send_response(self, code):
            captured["code"] = code

        def end_headers(self):
            pass

        def send_header(self, k, v):
            captured.setdefault("headers", {})[k] = v

    return H(), captured


@pytest.fixture
def reasoning_route_env(tmp_path, monkeypatch):
    """Point routes.SESSION_DIR at a temp dir and disable auth for handle_get."""
    from api import models, routes
    import api.auth as auth
    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(routes, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(auth, "is_auth_enabled", lambda: False)
    return tmp_path


def test_reasoning_endpoint_reads_side_store(reasoning_route_env, monkeypatch):
    from api import routes, models
    tmp_path = reasoning_route_env
    sid = "s-ep"
    msgs = [
        _msg("user", "hi"),
        _msg("assistant", "a", "the-stored-cot"),
        _msg("assistant", "b", "second-cot"),
    ]
    _make_sidecar(tmp_path, sid, msgs)
    cot_store.write_full(tmp_path, sid, msgs)  # as Session.load would
    monkeypatch.setattr(
        routes, "_session_id_visible_to_request_profile",
        lambda handler, sid, *, emit_error=True: True,
    )

    from urllib.parse import ParseResult
    h, captured = _make_handler()
    parsed = ParseResult(
        scheme="", netloc="", path="/api/session/reasoning",
        query=f"session_id={sid}&indices=1,2",
        params="", fragment="",
    )
    ok = routes.handle_get(h, parsed)
    assert ok
    assert captured["code"] == 200
    body = json.loads(captured["body"])
    assert body["reasoning"] == {"1": "the-stored-cot", "2": "second-cot"}


def test_copy_store_migrates_cot_across_session_ids():
    """Duplicate/fork path: CoT text follows messages moved to a new id.

    Regression guard for the /api/session/duplicate gap — messages carrying
    _has_cot markers reference records in the SOURCE session's store; without
    copy_store the destination session's thinking cards 404 on expand.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src, dst = "src-round", "dst-round"
        n = cot_store.write_full(d, src, [
            _msg("user", "q"),
            _msg("assistant", "", "carried thought"),
            _msg("assistant", "a"),
        ])
        assert n == 3
        assert cot_store.read_records(d, src, [1]) == {1: "carried thought"}
        # Before the copy the destination store has no records.
        assert cot_store.read_records(d, dst, [1]) == {}
        assert cot_store.copy_store(d, src, dst) is True
        assert cot_store.read_records(d, dst, [1]) == {1: "carried thought"}
        # Source store is untouched (a copy, not a move).
        assert cot_store.read_records(d, src, [1]) == {1: "carried thought"}
        # Copying a store that does not exist is a clean no-op.
        assert cot_store.copy_store(d, "absent", "empty") is False


def test_reasoning_endpoint_requires_session_id(reasoning_route_env):
    from api import routes
    from urllib.parse import ParseResult
    h, captured = _make_handler()
    parsed = ParseResult(scheme="", netloc="", path="/api/session/reasoning",
                         query="", params="", fragment="")
    routes.handle_get(h, parsed)
    assert captured["code"] == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
