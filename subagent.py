"""Threaded sub-agent system for spawning nested agent loops."""
from __future__ import annotations

import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubAgentTask:
    """Represents a sub-agent task with lifecycle tracking."""
    id: str
    prompt: str
    status: str = "pending"       # pending | running | completed | failed | cancelled
    result: Optional[str] = None
    depth: int = 0
    _cancel_flag: bool = False
    _future: Optional[Future] = field(default=None, repr=False)


def _agent_run(prompt, state, config, system_prompt, depth=0, cancel_check=None):
    """Lazy-import wrapper to avoid circular dependency with agent module."""
    from agent import run
    return run(prompt, state, config, system_prompt, depth=depth, cancel_check=cancel_check)


def _extract_final_text(messages):
    """Walk backwards through messages, return first assistant content string."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


class SubAgentManager:
    """Manages concurrent sub-agent tasks using a thread pool."""

    def __init__(self, max_concurrent: int = 3, max_depth: int = 3):
        self.tasks: Dict[str, SubAgentTask] = {}
        self.max_concurrent = max_concurrent
        self.max_depth = max_depth
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent)

    def spawn(self, prompt: str, config: dict, system_prompt: str, depth: int = 0) -> SubAgentTask:
        """Spawn a new sub-agent task.

        Args:
            prompt: user message for the sub-agent
            config: agent configuration dict
            system_prompt: system prompt for the sub-agent
            depth: current nesting depth

        Returns:
            SubAgentTask tracking the spawned work
        """
        task_id = uuid.uuid4().hex[:12]
        task = SubAgentTask(id=task_id, prompt=prompt, depth=depth)

        if depth >= self.max_depth:
            task.status = "failed"
            task.result = f"Max depth ({self.max_depth}) exceeded"
            self.tasks[task_id] = task
            return task

        self.tasks[task_id] = task

        def _run():
            from agent import AgentState
            task.status = "running"
            try:
                state = AgentState()
                gen = _agent_run(
                    prompt, state, config, system_prompt,
                    depth=depth + 1,
                    cancel_check=lambda: task._cancel_flag,
                )
                # Drain the generator to completion
                for _event in gen:
                    if task._cancel_flag:
                        break

                if task._cancel_flag:
                    task.status = "cancelled"
                    task.result = None
                else:
                    task.result = _extract_final_text(state.messages)
                    task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.result = f"Error: {e}"

        task._future = self._pool.submit(_run)
        return task

    def wait(self, task_id: str, timeout: float = None) -> Optional[SubAgentTask]:
        """Block until a task completes or timeout expires.

        Returns:
            The task, or None if task_id is unknown.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None
        if task._future is not None:
            try:
                task._future.result(timeout=timeout)
            except Exception:
                pass  # timeout or other error — task status already set by _run
        return task

    def get_result(self, task_id: str) -> Optional[str]:
        """Return the result string for a completed task, or None."""
        task = self.tasks.get(task_id)
        if task is None:
            return None
        return task.result

    def list_tasks(self) -> List[SubAgentTask]:
        """Return all tracked tasks."""
        return list(self.tasks.values())

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task.

        Returns:
            True if the cancel flag was set, False if task not found or not running.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status == "running":
            task._cancel_flag = True
            return True
        return False

    def shutdown(self):
        """Cancel all running tasks and shut down the thread pool."""
        for task in self.tasks.values():
            if task.status == "running":
                task._cancel_flag = True
        self._pool.shutdown(wait=True)
