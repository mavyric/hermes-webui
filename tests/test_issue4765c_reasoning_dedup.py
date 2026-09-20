"""Regression tests for CoT dedup at rest (#4765 follow-up, Patch A).

The webui sidecar triple-stored each assistant message's chain-of-thought:
``reasoning`` (display-canonical), ``reasoning_content`` (provider alias,
byte-identical in 99.8% of messages), and the CoT re-embedded inside the
``api_content`` request envelope (an agent-sidecar field the webui never
reads). On a 261k-message inference session that is ~1.8 GB of a ~0.7 MB
visible conversation and ~7 GiB of parse-time RAM.

Patch A stores the CoT ONCE:
  * identical ``reasoning``/``reasoning_content`` -> drop the alias;
  * only ``reasoning_content`` present -> canonicalize to ``reasoning``;
  * they differ -> keep both (lossless);
  * ``api_content`` -> dropped from the sidecar (agent keeps its own copy in
    state.db for prompt-cache stability);
  * provider replay: ``_sanitize_messages_for_api`` projects
    ``reasoning_content`` from ``reasoning`` so the model-facing history is
    identical to the pre-dedup layout.

Invariants proven here:
  1. dedup rules (drop / canonicalize / keep-both / drop api_content);
  2. idempotence (second pass is a no-op);
  3. non-assistant and non-reasoning messages are untouched;
  4. save() persists the slimmed sidecar (alias + api_content absent on disk,
     reasoning present, message_count unchanged);
  5. load() self-heals a pre-existing bloated sidecar on first read;
  6. provider replay projection restores ``reasoning_content`` in the
     sanitized, model-facing message.
"""
import json
import time

import pytest


@pytest.fixture
def isolated_session_env():
    """Isolate all SESSIONS-cache global state onto a throwaway temp dir."""
    from api import config as _cfg
    from api import models as _models

    import collections
    import shutil
    import tempfile
    import threading
    from pathlib import Path

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


def _asst(reasoning=None, reasoning_content=None, content="hi", api_content=None):
    m = {"role": "assistant", "content": content, "timestamp": time.time()}
    if reasoning is not None:
        m["reasoning"] = reasoning
    if reasoning_content is not None:
        m["reasoning_content"] = reasoning_content
    if api_content is not None:
        m["api_content"] = api_content
    return m


# ─────────────────────────── dedup rules ────────────────────────────────────

def test_identical_reasoning_alias_is_dropped():
    from api.models import _dedupe_session_message_reasoning

    msgs = [_asst(reasoning="think", reasoning_content="think")]
    out, changed = _dedupe_session_message_reasoning(msgs)
    assert changed is True
    assert out[0]["reasoning"] == "think"
    assert "reasoning_content" not in out[0]


def test_only_reasoning_content_is_canonicalized():
    from api.models import _dedupe_session_message_reasoning

    msgs = [_asst(reasoning_content="think")]
    out, changed = _dedupe_session_message_reasoning(msgs)
    assert changed is True
    assert out[0]["reasoning"] == "think"
    assert "reasoning_content" not in out[0]


def test_differing_reasoning_keeps_both():
    """When the two copies genuinely differ, NEVER drop the provider copy."""
    from api.models import _dedupe_session_message_reasoning

    msgs = [_asst(reasoning="display", reasoning_content="provider-verbatim")]
    out, changed = _dedupe_session_message_reasoning(msgs)
    assert out[0]["reasoning"] == "display"
    assert out[0]["reasoning_content"] == "provider-verbatim"
    # changed may be True only if api_content existed; here it did not.
    assert changed is False


def test_api_content_is_dropped():
    from api.models import _dedupe_session_message_reasoning

    msgs = [_asst(reasoning="x", api_content="request envelope")]
    out, changed = _dedupe_session_message_reasoning(msgs)
    assert changed is True
    assert "api_content" not in out[0]
    assert out[0]["reasoning"] == "x"


def test_non_assistant_and_plain_messages_untouched():
    from api.models import _dedupe_session_message_reasoning

    user = {"role": "user", "content": "q", "timestamp": time.time()}
    tool = {"role": "tool", "content": "r", "timestamp": time.time(), "api_content": "keep"}
    plain = _asst(content="answer")  # no reasoning at all
    msgs = [user, tool, plain]
    out, changed = _dedupe_session_message_reasoning(msgs)
    assert out[0] is user
    assert out[1] is tool and "api_content" in tool  # tool rows are never touched
    assert out[2] is plain
    assert changed is False


def test_dedup_is_idempotent():
    from api.models import _dedupe_session_message_reasoning

    msgs = [_asst(reasoning="x", reasoning_content="x", api_content="env")]
    _dedupe_session_message_reasoning(msgs)
    _out2, changed2 = _dedupe_session_message_reasoning(msgs)
    assert changed2 is False
    assert _out2[0]["reasoning"] == "x"


# ─────────────────────── save() persists the slim form ──────────────────────

def test_save_persists_reasoning_once(isolated_session_env):
    """The on-disk sidecar must carry the CoT exactly once (#4765 Patch A)."""
    from api.models import Session

    s = Session(
        session_id="cotdedup01",
        title="cot",
        messages=[
            {"role": "user", "content": "q", "timestamp": time.time()},
            _asst(reasoning="think A", reasoning_content="think A",
                  api_content="big request envelope", content="answer"),
        ],
    )
    s.save()

    raw = json.loads(s.path.read_text(encoding="utf-8"))
    msgs = raw["messages"]
    asst = [m for m in msgs if m.get("role") == "assistant"][0]
    assert asst["reasoning"] == "think A"
    assert "reasoning_content" not in asst, "duplicate CoT alias survived on disk"
    assert "api_content" not in asst, "api_content sidecar survived on disk"
    # message_count must be preserved (the #1558 shrink guard keys off it).
    assert raw["message_count"] == 2
    # No shrink backup should have been produced (same message count).
    assert not s.path.with_suffix(".json.bak").exists()


# ─────────────────────── load() self-heals old files ────────────────────────

def test_load_self_heals_bloated_sidecar(isolated_session_env):
    """A pre-existing sidecar with the triple-stored CoT is slimmed on read."""
    from api.models import Session

    n = 50
    bloated = {
        "session_id": "bloated001",
        "title": "bloated",
        "message_count": n,
        "messages": [
            {"role": "user", "content": "q", "timestamp": time.time()},
            *[
                _asst(reasoning=f"think {i}", reasoning_content=f"think {i}",
                      api_content=f"envelope {i}")
                for i in range(n - 1)
            ],
        ],
    }
    path = isolated_session_env / "bloated001.json"
    before = len(json.dumps(bloated, ensure_ascii=False))
    path.write_text(json.dumps(bloated, ensure_ascii=False), encoding="utf-8")

    loaded = Session.load("bloated001")
    assert loaded is not None
    assert len(loaded.messages) == n

    raw = json.loads(path.read_text(encoding="utf-8"))
    after = len(json.dumps(raw, ensure_ascii=False))
    assert after < before, "sidecar did not shrink after self-heal"
    from api import cot_store
    for m in raw["messages"]:
        if m.get("role") == "assistant":
            assert "reasoning_content" not in m
            assert "api_content" not in m
            # #4765 follow-up: CoT text now lives in the side store, not inline.
            assert "reasoning" not in m
            assert m["_has_cot"] is True
    # Side store carries the preserved CoT verbatim.
    recs = cot_store.read_records(isolated_session_env, "bloated001", list(range(1, n)))
    assert recs[1] == "think 0" and recs[n - 1] == f"think {n - 2}"
    # .bak is allowed (intentional shrink) and must retain the pre-shrink text.
    bak = path.with_suffix(".json.bak")
    if bak.exists():
        bak_msgs = json.loads(bak.read_text(encoding="utf-8")).get("messages") or []
        assert len(bak_msgs) == n
        assert any("reasoning_content" in m for m in bak_msgs)


# ─────────────────── provider replay projection (sibling-safe) ──────────────

def test_provider_replay_projection_restores_alias():
    """_sanitize_messages_for_api must re-expose reasoning_content for replay.

    After dedup the sidecar stores CoT only in ``reasoning``; providers that
    replay historical CoT read ``reasoning_content``. The projection keeps the
    model-facing history identical to the pre-dedup layout.
    """
    from api.streaming import _sanitize_messages_for_api

    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer", "reasoning": "think"},
    ]
    out = _sanitize_messages_for_api(msgs)
    asst = [m for m in out if m.get("role") == "assistant"][0]
    assert asst.get("reasoning_content") == "think", (
        "provider-facing reasoning_content was not projected from reasoning — "
        "replay providers would see the CoT disappear after dedup"
    )
    # The display-only field must not leak into the API-safe set.
    assert "reasoning" not in asst


def test_provider_replay_projection_skips_when_alias_present():
    """An explicit (differing) alias is preserved verbatim, not overwritten."""
    from api.streaming import _sanitize_messages_for_api

    msgs = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "a",
            "reasoning": "display",
            "reasoning_content": "provider-verbatim",
        },
    ]
    out = _sanitize_messages_for_api(msgs)
    asst = [m for m in out if m.get("role") == "assistant"][0]
    assert asst["reasoning_content"] == "provider-verbatim"


def test_provider_replay_projection_skips_empty_reasoning():
    from api.streaming import _sanitize_messages_for_api

    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "   "},
    ]
    out = _sanitize_messages_for_api(msgs)
    asst = [m for m in out if m.get("role") == "assistant"][0]
    assert "reasoning_content" not in asst
