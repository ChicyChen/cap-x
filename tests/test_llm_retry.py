"""LLM retry behaviour on provider outages.

An upstream Bedrock 503 stalled every real-robot episode: the original loop
retried FOREVER with 150-330s blind sleeps, so the operator saw silence and
concluded the server had hung. Retries must be bounded, short, visible, and end
in a loud failure.
"""

from __future__ import annotations

import types

import pytest


class _Resp:
    def __init__(self, code, text="boom"):
        self.status_code = code
        self.text = text

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}

    def raise_for_status(self):
        return None


def _args():
    return types.SimpleNamespace(
        model="aws/anthropic/bedrock-claude-opus-4-6",
        api_key="k",
        server_url="https://example/v1/chat/completions",
        temperature=0.0,
        max_tokens=256,
        reasoning_effort=None,
        debug=False,
    )


class TestBoundedRetries:
    def test_gives_up_and_raises(self, monkeypatch):
        import capx.llm.client as c

        monkeypatch.setenv("CAPX_LLM_MAX_RETRIES", "2")
        monkeypatch.setenv("CAPX_LLM_RETRY_SLEEP_S", "0")
        monkeypatch.setattr(c.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def post(*a, **kw):
            calls["n"] += 1
            return _Resp(503)

        monkeypatch.setattr(c.requests, "post", post)
        with pytest.raises(RuntimeError, match="after 2 retries"):
            c.query_model(_args(), [{"role": "user", "content": "hi"}])
        assert calls["n"] == 3, "1 initial + 2 retries, not unbounded"

    def test_recovers_when_the_provider_comes_back(self, monkeypatch):
        import capx.llm.client as c

        monkeypatch.setenv("CAPX_LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("CAPX_LLM_RETRY_SLEEP_S", "0")
        monkeypatch.setattr(c.time, "sleep", lambda s: None)
        seq = [_Resp(503), _Resp(503), _Resp(200)]

        monkeypatch.setattr(c.requests, "post", lambda *a, **kw: seq.pop(0))
        out = c.query_model(_args(), [{"role": "user", "content": "hi"}])
        assert out is not None

    def test_backoff_is_capped(self, monkeypatch):
        """150-330s sleeps are unusable in an interactive robot session."""
        import capx.llm.client as c

        monkeypatch.setenv("CAPX_LLM_MAX_RETRIES", "8")
        monkeypatch.setenv("CAPX_LLM_RETRY_SLEEP_S", "20")
        slept: list[float] = []
        monkeypatch.setattr(c.time, "sleep", lambda s: slept.append(s))
        monkeypatch.setattr(c.requests, "post", lambda *a, **kw: _Resp(503))
        with pytest.raises(RuntimeError):
            c.query_model(_args(), [{"role": "user", "content": "hi"}])
        assert slept, "should have slept"
        assert max(slept) <= 65, f"backoff not capped: {max(slept)}"

    def test_outage_is_published_to_the_dashboard(self, monkeypatch):
        import capx.llm.client as c
        from capx.monitor.events import BUS, EventKind

        BUS._history.clear()
        monkeypatch.setenv("CAPX_LLM_MAX_RETRIES", "1")
        monkeypatch.setenv("CAPX_LLM_RETRY_SLEEP_S", "0")
        monkeypatch.setattr(c.time, "sleep", lambda s: None)
        monkeypatch.setattr(c.requests, "post", lambda *a, **kw: _Resp(503))
        with pytest.raises(RuntimeError):
            c.query_model(_args(), [{"role": "user", "content": "hi"}])
        errs = [e for e in BUS._history if e.kind == EventKind.ERROR]
        assert errs, "a provider outage must be visible, not silent"
        assert "provider unavailable" in errs[-1].text


class TestConfigUsesTheWorkingModel:
    def test_differencing_model_matches_the_cli_model(self):
        import pathlib

        cfg = pathlib.Path("env_configs/real/real_franky.yaml").read_text()
        line = [
            l for l in cfg.splitlines()
            if l.startswith("visual_differencing_model:")
        ][0]
        assert "opus-4-6" in line, (
            "use_img_differencing is true, so this model is called once per "
            "trial BEFORE code generation; a dead model here stalls every "
            f"episode. Got: {line}"
        )


class TestDifferencingModelIsAcceptedByTheAssertion:
    """trial.py:699 asserts the differencing model is in VLM_MODELS.

    Switching visual_differencing_model to opus-4-6 (because 4-7 was 503-ing)
    made CaP-X crash on the FIRST driver frame with

        AssertionError: Image/video differencing model must be in the list of
                        VLM models

    which the driver saw only as 'no close frame received or sent'. The config
    and the model registry must agree.
    """

    def test_the_configured_model_is_registered(self):
        import pathlib
        import re

        from capx.llm.client import VLM_MODELS

        cfg = pathlib.Path("env_configs/real/real_franky.yaml").read_text()
        line = [
            l for l in cfg.splitlines()
            if l.startswith("visual_differencing_model:")
        ][0]
        model = line.split(":", 1)[1].strip()
        assert model in VLM_MODELS, (
            f"{model} is configured for image differencing but is not in "
            f"VLM_MODELS, so trial.py:699 will assert on the first frame"
        )

    def test_img_differencing_is_actually_enabled(self):
        """If it were off, the assertion would not fire -- keep them in sync."""
        import pathlib

        cfg = pathlib.Path("env_configs/real/real_franky.yaml").read_text()
        assert "use_img_differencing: true" in cfg
