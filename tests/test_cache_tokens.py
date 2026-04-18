"""Tests for cache token tracking end-to-end."""
import dataclasses
import pytest


def test_assistant_turn_has_cache_fields():
    """AssistantTurn carries cache_read_tokens and cache_write_tokens."""
    from providers import AssistantTurn
    turn = AssistantTurn(
        text="hello", tool_calls=[], in_tokens=100, out_tokens=50,
        cache_read_tokens=80, cache_write_tokens=20,
    )
    assert turn.cache_read_tokens == 80
    assert turn.cache_write_tokens == 20


def test_assistant_turn_cache_defaults_zero():
    """Cache fields default to 0 for backward compat."""
    from providers import AssistantTurn
    turn = AssistantTurn(text="hi", tool_calls=[], in_tokens=10, out_tokens=5)
    assert turn.cache_read_tokens == 0
    assert turn.cache_write_tokens == 0


def test_agent_state_accumulates_cache_tokens():
    """AgentState accumulates cache tokens from AssistantTurn."""
    from agent import AgentState
    state = AgentState()
    assert state.total_cache_read_tokens == 0
    assert state.total_cache_write_tokens == 0

    # Simulate what the agent loop does
    state.total_cache_read_tokens += 80
    state.total_cache_write_tokens += 20
    state.total_cache_read_tokens += 60
    state.total_cache_write_tokens += 10

    assert state.total_cache_read_tokens == 140
    assert state.total_cache_write_tokens == 30


def test_checkpoint_snapshot_includes_cache():
    """make_snapshot persists cache tokens in token_snapshot."""
    from checkpoint.store import make_snapshot
    from agent import AgentState

    state = AgentState()
    state.total_input_tokens = 500
    state.total_output_tokens = 200
    state.total_cache_read_tokens = 300
    state.total_cache_write_tokens = 50
    state.turn_count = 3
    state.messages = [{"role": "user", "content": "test"}]

    snapshot = make_snapshot(state, "test-session", "hello user")
    assert snapshot.token_snapshot["cache_read"] == 300
    assert snapshot.token_snapshot["cache_write"] == 50
    assert snapshot.token_snapshot["input"] == 500
    assert snapshot.token_snapshot["output"] == 200
