"""
agent_runner.py — Autonomous agent loop driven by task templates.

Design
------
* Each AgentRunner owns an isolated AgentState (separate from the main REPL).
* Templates are Markdown files (built-ins in agent_templates/ or user-supplied
  path) describing what the agent should do, inspired by Karpathy's autoresearch
  program.md pattern.
* The loop calls agent.run() for each iteration, draining the generator.
  PermissionRequests are auto-granted (autonomous mode) with a notification.
* After each iteration a ≤500-char summary is sent via send_fn (bridge / terminal).
* Iteration history is persisted to ~/.cheetahclaws/agents/<name>/log.jsonl.
* call stop() or send_fn receives "!agent-stop" to terminate the loop.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import logging_utils as _log

# ── Template resolution ────────────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).parent / "agent_templates"
_USER_TEMPLATES_DIR = Path.home() / ".cheetahclaws" / "agent_templates"


def list_templates() -> list[dict]:
    """Return all known templates (built-in + user-defined)."""
    result = []
    for d, source in [(_TEMPLATES_DIR, "built-in"), (_USER_TEMPLATES_DIR, "user")]:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.md")):
            result.append({"name": f.stem, "source": source, "path": str(f)})
    return result


def load_template(name_or_path: str) -> tuple[str, str]:
    """Load a template by name or file path.

    Returns (template_content, resolved_path).
    Raises FileNotFoundError if not found.
    """
    p = Path(name_or_path)
    if p.exists():
        return p.read_text(encoding="utf-8"), str(p)

    # Search built-in then user
    for d in [_USER_TEMPLATES_DIR, _TEMPLATES_DIR]:
        candidate = d / f"{name_or_path}.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8"), str(candidate)

    available = [t["name"] for t in list_templates()]
    raise FileNotFoundError(
        f"Template '{name_or_path}' not found. "
        f"Available: {', '.join(available) or '(none)'}"
    )


# ── Registry ───────────────────────────────────────────────────────────────

_runners: dict[str, "AgentRunner"] = {}
_runners_lock = threading.Lock()


def get_runner(name: str) -> "AgentRunner | None":
    with _runners_lock:
        r = _runners.get(name)
        if r and not r.is_alive:
            _runners.pop(name, None)
            return None
        return r


def list_runners() -> list["AgentRunner"]:
    with _runners_lock:
        return list(_runners.values())


def start_runner(
    name: str,
    template_name: str,
    args: str,
    config: dict,
    send_fn: Optional[Callable[[str], None]] = None,
    interval: float = 2.0,
    auto_approve: bool = True,
) -> "AgentRunner":
    """Create and start an AgentRunner; kill any previous runner with same name."""
    template_content, template_path = load_template(template_name)
    runner = AgentRunner(
        name=name,
        template_content=template_content,
        template_path=template_path,
        args=args,
        config=config,
        send_fn=send_fn,
        interval=interval,
        auto_approve=auto_approve,
    )
    with _runners_lock:
        old = _runners.get(name)
        if old:
            old.stop()
        _runners[name] = runner
    runner.start()
    return runner


def stop_runner(name: str) -> bool:
    with _runners_lock:
        r = _runners.pop(name, None)
    if r:
        r.stop()
        return True
    return False


def stop_all() -> int:
    with _runners_lock:
        runners = list(_runners.values())
        _runners.clear()
    for r in runners:
        r.stop()
    return len(runners)


# ── AgentRunner ────────────────────────────────────────────────────────────

_LOG_DIR = Path.home() / ".cheetahclaws" / "agents"


@dataclass
class _IterationRecord:
    iteration: int
    timestamp: str
    summary: str
    status: str  # "ok" | "error" | "permission"
    duration_s: float


class AgentRunner:
    """Runs an autonomous agent loop driven by a task template."""

    def __init__(
        self,
        name: str,
        template_content: str,
        template_path: str,
        args: str,
        config: dict,
        send_fn: Optional[Callable[[str], None]],
        interval: float = 2.0,
        auto_approve: bool = True,
    ) -> None:
        self.name = name
        self.template = template_content
        self.template_path = template_path
        self.args = args
        self._config = config.copy()
        self.send_fn = send_fn
        self.interval = interval
        self.auto_approve = auto_approve

        self.iteration = 0
        self.status = "idle"
        self._stop_event = threading.Event()
        self._history: list[_IterationRecord] = []
        self._thread: threading.Thread | None = None
        self._log_dir = _LOG_DIR / name
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ── Public interface ───────────────────────────────────────────────────

    def start(self) -> None:
        self.status = "starting"
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True,
            name=f"agent-{self.name}",
        )
        self._thread.start()
        _log.info("agent_runner_start", name=self.name,
                  template=self.template_path, args=self.args[:100])

    def stop(self) -> None:
        self._stop_event.set()
        self.status = "stopping"
        _log.info("agent_runner_stop", name=self.name, iteration=self.iteration)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def recent_log(self, n: int = 5) -> list[_IterationRecord]:
        return self._history[-n:]

    def summary_text(self) -> str:
        lines = [f"Agent: {self.name}  status={self.status}  iter={self.iteration}"]
        for rec in self.recent_log(3):
            lines.append(f"  [{rec.iteration}] {rec.status} ({rec.duration_s:.1f}s): {rec.summary[:120]}")
        return "\n".join(lines)

    # ── Internal loop ──────────────────────────────────────────────────────

    def _notify(self, text: str) -> None:
        """Send a message to the phone/terminal."""
        if self.send_fn:
            try:
                self.send_fn(text)
            except Exception:
                pass
        else:
            print(text)

    def _run_loop(self) -> None:
        from agent import AgentState, PermissionRequest, TurnDone
        from agent import TextChunk, ToolStart, ToolEnd

        state = AgentState()
        config = self._config.copy()
        config["_auto_agent"] = True
        config["_auto_approve"] = self.auto_approve

        system_prompt = (
            "You are an autonomous agent executing the following task program. "
            "Run it faithfully and autonomously. After completing each iteration, "
            "write a brief 1-2 sentence summary of what you did and what you'll do next.\n\n"
            f"=== TASK PROGRAM ===\n{self.template}\n=== END PROGRAM ==="
        )

        self.status = "running"
        self._notify(
            f"🚀 Agent **{self.name}** started.\n"
            f"Template: `{Path(self.template_path).name}`\n"
            f"Args: {self.args or '(none)'}\n"
            f"Auto-approve: {self.auto_approve}\n"
            "Send `!agent stop {name}` to stop."
        )

        iteration = 0
        while not self._stop_event.is_set():
            iteration += 1
            self.iteration = iteration
            self.status = f"running (iter {iteration})"
            t_start = time.monotonic()

            prompt = (
                f"Begin the program. Args: {self.args}" if iteration == 1 and self.args
                else "Begin the program." if iteration == 1
                else "Continue to the next iteration of the program."
            )

            text_chunks: list[str] = []
            rec_status = "ok"

            try:
                for event in __import__("agent").run(
                    prompt, state, config, system_prompt
                ):
                    if self._stop_event.is_set():
                        break

                    if isinstance(event, TextChunk):
                        text_chunks.append(event.text)

                    elif isinstance(event, PermissionRequest):
                        if self.auto_approve:
                            event.granted = True
                            self._notify(
                                f"🔐 [{self.name}] Auto-approved: {event.description[:120]}"
                            )
                            rec_status = "permission"
                        else:
                            self._notify(
                                f"🔐 [{self.name}] Permission needed (agent paused):\n"
                                f"{event.description}\n\n"
                                "The agent cannot continue without approval. "
                                "Restart with `--auto-approve` to enable autonomous mode."
                            )
                            event.granted = False
                            self._stop_event.set()
                            break

                    elif isinstance(event, ToolStart):
                        cmd_preview = str(
                            (event.inputs or {}).get("command",
                             (event.inputs or {}).get("file_path", ""))
                        ).strip()[:60]
                        _log.debug("agent_tool_start", name=self.name,
                                   tool=event.name, cmd=cmd_preview)

            except Exception as exc:
                rec_status = "error"
                err_msg = str(exc)[:300]
                text_chunks.append(f"\n[ERROR: {err_msg}]")
                self._notify(f"⚠ [{self.name}] iter {iteration} error:\n{err_msg}")
                _log.warn("agent_runner_error", name=self.name, iteration=iteration,
                          error=err_msg)
                # Brief pause before retrying
                self._stop_event.wait(10.0)

            duration = time.monotonic() - t_start
            summary = "".join(text_chunks).strip()[-400:] or "(no output)"

            rec = _IterationRecord(
                iteration=iteration,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                summary=summary[:400],
                status=rec_status,
                duration_s=round(duration, 1),
            )
            self._history.append(rec)
            self._persist_record(rec)

            # Report iteration result
            if rec_status != "error":
                self._notify(
                    f"✅ [{self.name}] iter {iteration} ({duration:.0f}s):\n"
                    f"{summary[:400]}"
                )

            _log.info("agent_runner_iter", name=self.name, iteration=iteration,
                      status=rec_status, duration_s=rec.duration_s)

            # Wait before next iteration (stop event wakes it early)
            self._stop_event.wait(self.interval)

        self.status = "stopped"
        self._notify(f"⏹ Agent **{self.name}** stopped after {iteration} iterations.")
        _log.info("agent_runner_stopped", name=self.name, iterations=iteration)

    def _persist_record(self, rec: _IterationRecord) -> None:
        log_file = self._log_dir / "log.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "iteration": rec.iteration,
                    "timestamp": rec.timestamp,
                    "status": rec.status,
                    "duration_s": rec.duration_s,
                    "summary": rec.summary,
                }) + "\n")
        except Exception:
            pass
