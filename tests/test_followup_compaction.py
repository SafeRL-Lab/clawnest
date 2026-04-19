"""Tests for followup_compaction module."""
import pytest

from followup_compaction import (
    compact_tool_history, _build_tool_call_lookup, _build_stub,
    _input_brief, _escape_xml_attr,
    DEFAULT_EXEMPT_TOOLS,
)


class TestCompactToolHistory:
    def _make_messages(self):
        return [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "tc1", "name": "Read", "input": {"file_path": "/a.py"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "name": "Read", "content": "file contents..."},
            {"role": "user", "content": "turn 2"},
            {"role": "assistant", "content": "done"},
        ]

    def test_stubs_old_tool_results(self):
        msgs = self._make_messages()
        result = compact_tool_history(msgs)
        assert "<tool_use_elided" in result[2]["content"]
        assert result[4]["content"] == "done"

    def test_preserves_current_turn(self):
        msgs = self._make_messages()
        result = compact_tool_history(msgs)
        assert result[3]["content"] == "turn 2"
        assert result[4]["content"] == "done"

    def test_no_compaction_if_single_turn(self):
        msgs = [
            {"role": "user", "content": "only turn"},
            {"role": "assistant", "content": "reply"},
        ]
        result = compact_tool_history(msgs)
        assert result[1]["content"] == "reply"

    def test_exempt_tools_preserved(self):
        msgs = [
            {"role": "user", "content": "t1"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "tc1", "name": "Write", "input": {"file_path": "/a.py", "content": "x"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "name": "Write", "content": "Written."},
            {"role": "user", "content": "t2"},
        ]
        result = compact_tool_history(msgs)
        assert result[2]["content"] == "Written."

    def test_keep_last_n_turns(self):
        msgs = [
            {"role": "user", "content": "t1"},
            {"role": "tool", "tool_call_id": "tc1", "name": "Read", "content": "data1"},
            {"role": "user", "content": "t2"},
            {"role": "tool", "tool_call_id": "tc2", "name": "Read", "content": "data2"},
            {"role": "user", "content": "t3"},
        ]
        result = compact_tool_history(msgs, keep_last_n_turns=1)
        assert result[3]["content"] == "data2"


class TestBuildStub:
    def test_read_stub(self):
        stub = _build_stub("Read", {"file_path": "/a.py"})
        assert "tool_use_elided" in stub
        assert "Read" in stub
        assert "/a.py" in stub

    def test_bash_stub_truncates(self):
        long_cmd = "x" * 200
        stub = _build_stub("Bash", {"command": long_cmd})
        assert "..." in stub

    def test_grep_stub(self):
        stub = _build_stub("Grep", {"pattern": "TODO", "path": "/src"})
        assert "TODO" in stub
        assert "/src" in stub

    def test_glob_stub(self):
        stub = _build_stub("Glob", {"pattern": "**/*.py"})
        assert "**/*.py" in stub

    def test_generic_stub(self):
        stub = _build_stub("Custom", {"key": "val"})
        assert "Custom" in stub


class TestInputBrief:
    def test_read(self):
        assert "file_path=/a.py" in _input_brief("Read", {"file_path": "/a.py"})

    def test_read_with_offset(self):
        brief = _input_brief("Read", {"file_path": "/a.py", "offset": 10, "limit": 20})
        assert "offset=10" in brief
        assert "limit=20" in brief

    def test_bash(self):
        brief = _input_brief("Bash", {"command": "ls -la"})
        assert "ls -la" in brief

    def test_long_generic(self):
        inp = {"key": "x" * 200}
        brief = _input_brief("Unknown", inp)
        assert len(brief) <= 120


class TestEscapeXmlAttr:
    def test_ampersand(self):
        assert "&amp;" in _escape_xml_attr("a&b")

    def test_lt_gt(self):
        assert "&lt;" in _escape_xml_attr("<")
        assert "&gt;" in _escape_xml_attr(">")

    def test_quote(self):
        assert "&quot;" in _escape_xml_attr('"hello"')


class TestBuildToolCallLookup:
    def test_builds_lookup(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "tc1", "name": "Read", "input": {"file_path": "/x"}},
                {"id": "tc2", "name": "Bash", "input": {"command": "ls"}},
            ]},
        ]
        lookup = _build_tool_call_lookup(msgs)
        assert lookup["tc1"] == ("Read", {"file_path": "/x"})
        assert lookup["tc2"] == ("Bash", {"command": "ls"})

    def test_skips_non_assistant(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _build_tool_call_lookup(msgs) == {}
