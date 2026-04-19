"""End-to-end: drive a real agent.run() conversation where the LLM calls
ContextGC, and verify gc_state ends up correctly populated + survives a
session save/load roundtrip.

Only the LLM provider is mocked (via monkeypatching agent.stream). The tool
registry, session serializer and ContextGC dispatch all run for real.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools as _tools_init  # noqa: F401 - force built-in tool registration
from agent import AgentState, run
from providers import AssistantTurn, TextChunk
from tool_registry import ToolDef, register_tool
from commands.session import _build_session_data, _restore_state_from_data


def _scripted_stream(turns):
    """Yield pre-scripted AssistantTurn objects one per call to stream(...).

    Signature matches providers.stream(**kwargs). We ignore all kwargs.
    """
    cursor = iter(turns)

    def fake_stream(**_kwargs):
        spec = next(cursor)
        if spec.get("text"):
            yield TextChunk(spec["text"])
        yield AssistantTurn(
            text=spec.get("text", ""),
            tool_calls=spec.get("tool_calls") or [],
            in_tokens=1,
            out_tokens=1,
        )

    return fake_stream


@pytest.fixture
def echo_tool():
    """Register a simple echo tool that returns its input verbatim.

    Non-invasive: leaves the rest of the registry intact (built-ins + plugins
    loaded at module import) so unrelated tests sharing the process still see
    their tools. Only the echo entry is removed on teardown.
    """
    from tool_registry import _registry  # private, but fine for test isolation
    had_echo_before = "echo" in _registry
    register_tool(ToolDef(
        name="echo",
        schema={
            "name": "echo",
            "description": "echo",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
        func=lambda params, _cfg: f"echoed: {params.get('text', '')}",
        read_only=True, concurrent_safe=True,
    ))
    yield
    if not had_echo_before:
        _registry.pop("echo", None)


def test_llm_trashes_tool_result_via_contextgc_end_to_end(monkeypatch, echo_tool):
    """LLM calls echo, then ContextGC(trash=[echo_id]); gc_state is mutated."""
    turns = [
        # Turn 1 (first stream call): LLM issues the echo tool call.
        {"tool_calls": [
            {"id": "echo_42", "name": "echo", "input": {"text": "hi"}},
        ]},
        # Turn 2: LLM follows up with ContextGC to trash echo_42.
        {"tool_calls": [
            {"id": "gc_1", "name": "ContextGC", "input": {"trash": ["echo_42"]}},
        ]},
        # Turn 3: LLM emits plain text; no tool_calls → loop exits.
        {"text": "all set"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    config = {"model": "test", "permission_mode": "accept-all", "_session_id": "gc_e2e"}

    list(run("please echo and clean up", state, config, "system prompt"))

    assert state.gc_state.trashed_ids == {"echo_42"}
    assert "[ContextGC result]" not in state.gc_state.trashed_ids
    # Neither the ContextGC tool result nor the echo result are deleted from
    # state.messages -- only the OUTGOING messages on the next turn are reshaped.
    tool_results = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_results) == 2


def test_gc_state_survives_save_and_reload_via_session_helpers(monkeypatch, echo_tool, tmp_path):
    """Roundtrip through _build_session_data / _restore_state_from_data."""
    turns = [
        {"tool_calls": [
            {"id": "echo_1", "name": "echo", "input": {"text": "x"}},
        ]},
        {"tool_calls": [
            {"id": "gc_1", "name": "ContextGC", "input": {"trash": ["echo_1"]}},
        ]},
        {"text": "done"},
    ]
    monkeypatch.setattr("agent.stream", _scripted_stream(turns))

    state = AgentState()
    list(run("go", state, {"model": "test", "permission_mode": "accept-all",
                            "_session_id": "rt"}, "sys"))
    assert state.gc_state.trashed_ids == {"echo_1"}

    # Serialize to disk (through JSON to exercise the real save path).
    session_path: Path = tmp_path / "session.json"
    session_path.write_text(
        json.dumps(_build_session_data(state), default=str), encoding="utf-8"
    )

    # Restore into a brand-new state — trashed_ids must come back intact.
    reloaded = AgentState()
    _restore_state_from_data(
        reloaded, json.loads(session_path.read_text(encoding="utf-8"))
    )
    assert reloaded.gc_state.trashed_ids == {"echo_1"}


def test_disabled_tools_hides_contextgc_schema_from_llm(monkeypatch, echo_tool):
    """With config['disabled_tools']=['ContextGC'] the LLM never sees the schema."""
    captured_schemas = []

    def spy_stream(**kwargs):
        captured_schemas.append([s["name"] for s in kwargs.get("tool_schemas") or []])
        yield AssistantTurn(text="hello", tool_calls=[], in_tokens=1, out_tokens=1)

    monkeypatch.setattr("agent.stream", spy_stream)

    state = AgentState()
    list(run("hi", state, {
        "model": "test",
        "permission_mode": "accept-all",
        "_session_id": "gated",
        "disabled_tools": ["ContextGC"],
    }, "sys"))

    assert captured_schemas, "stream() must have been called at least once"
    for schemas in captured_schemas:
        assert "ContextGC" not in schemas
        assert "echo" in schemas  # non-disabled tool still present
