"""
commands/agent_cmd.py — /agent slash command for CheetahClaws.

  /agent                       → interactive wizard (beginner-friendly)
  /agent start <template> ...  → direct launch (power-user)
  /agent stop <name|all>       → stop agent(s)
  /agent list                  → show running agents
  /agent status <name>         → recent iteration log
  /agent templates             → list available templates

Direct-start options:
  --name <name>          Custom agent name (default: template name)
  --interval <seconds>   Pause between iterations (default: 2)
  --no-auto-approve      Pause for permissions instead of auto-granting

Examples:
  /agent                                    ← wizard
  /agent start research_assistant ~/papers/
  /agent start auto_bug_fixer --interval 5
  /agent start paper_writer outline.md
  /agent start auto_coder --task "add rate limiting"
  /agent start /path/to/custom.md
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

from ui.render import info, ok, warn, err, clr


# ── Wizard helpers ─────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    """Prompt with optional default; Ctrl-C → returns '' to signal abort."""
    suffix = f" [{clr(default, 'dim')}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        return "\x00"   # sentinel = cancelled


def _hdr(title: str) -> None:
    w = 56
    print("\n" + clr("╭" + "─" * w + "╮", "dim"))
    print(clr("│", "dim") + f"  {clr(title, 'cyan'):<54}" + clr("│", "dim"))
    print(clr("╰" + "─" * w + "╯", "dim") + "\n")


_MENU_ITEMS = [
    ("research_assistant", "📚", "Research Assistant",
     "Read papers → summarize → build related work"),
    ("auto_bug_fixer",     "🐛", "Auto Bug Fixer",
     "Run tests → find failures → fix & commit"),
    ("paper_writer",       "✍️ ", "Paper Writer",
     "Write paper sections from an outline"),
    ("auto_coder",         "💻", "Auto Coder",
     "Implement tasks from a backlog → test → commit"),
    ("__custom__",         "📄", "Custom template…",
     "Use your own .md program file"),
]


def _wizard(config: dict) -> bool:
    """Interactive wizard — returns True when done (start or abort)."""
    from agent_runner import list_templates, start_runner, get_runner

    # ── Menu ──────────────────────────────────────────────────────────────
    _hdr("🤖  Auto Agent  —  What do you want to do?")
    for i, (_, icon, label, desc) in enumerate(_MENU_ITEMS, 1):
        print(f"  {clr(str(i), 'yellow')}  {icon}  {clr(label, 'bold')}")
        print(f"        {clr(desc, 'dim')}\n")
    print(f"  {clr('q', 'yellow')}  Quit\n")

    choice_raw = _ask("Choice [1-5, q]", "")
    if choice_raw in ("\x00", "q", "Q", ""):
        info("Cancelled.")
        return True

    try:
        choice = int(choice_raw)
        if not 1 <= choice <= len(_MENU_ITEMS):
            raise ValueError
    except ValueError:
        err(f"Invalid choice '{choice_raw}'. Enter 1-{len(_MENU_ITEMS)} or q.")
        return True

    template_name, icon, label, _ = _MENU_ITEMS[choice - 1]

    # ── Per-template questions ────────────────────────────────────────────
    print()
    print(f"  {icon}  {clr(label, 'cyan')}\n")

    agent_args = ""
    agent_name = template_name if template_name != "__custom__" else "agent"
    interval   = 2.0
    auto_approve = True

    if template_name == "research_assistant":
        target = _ask("Paper directory or search topic", ".")
        if target == "\x00": info("Cancelled."); return True
        notes_out = _ask("Output notes file", "research_notes.md")
        if notes_out == "\x00": info("Cancelled."); return True
        agent_name = _ask("Agent name", "research")
        if agent_name == "\x00": info("Cancelled."); return True
        aa = _ask("Auto-approve file writes? [Y/n]", "Y")
        if aa == "\x00": info("Cancelled."); return True
        auto_approve = aa.strip().lower() not in ("n", "no")
        agent_args = f"{target} --output {notes_out}"

    elif template_name == "auto_bug_fixer":
        test_cmd = _ask("Test command", "pytest")
        if test_cmd == "\x00": info("Cancelled."); return True
        repo = _ask("Repo directory", ".")
        if repo == "\x00": info("Cancelled."); return True
        agent_name = _ask("Agent name", "bugfix")
        if agent_name == "\x00": info("Cancelled."); return True
        aa = _ask("Auto-approve shell commands? [Y/n]", "Y")
        if aa == "\x00": info("Cancelled."); return True
        auto_approve = aa.strip().lower() not in ("n", "no")
        agent_args = f"--test-cmd {shlex.quote(test_cmd)} --repo {shlex.quote(repo)}"

    elif template_name == "paper_writer":
        outline = _ask("Outline file path (required)", "")
        if outline in ("\x00", ""):
            err("Outline file is required.") if outline != "\x00" else info("Cancelled.")
            return True
        output = _ask("Output draft file", "paper_draft.md")
        if output == "\x00": info("Cancelled."); return True
        style = _ask("Writing style/venue", "NeurIPS")
        if style == "\x00": info("Cancelled."); return True
        agent_name = _ask("Agent name", "paper")
        if agent_name == "\x00": info("Cancelled."); return True
        auto_approve = True
        agent_args = f"{shlex.quote(outline)} --output {shlex.quote(output)} --style {shlex.quote(style)}"

    elif template_name == "auto_coder":
        task_input = _ask(
            "Task description  (or path to tasks.md)",
            "tasks.md",
        )
        if task_input == "\x00": info("Cancelled."); return True
        agent_name = _ask("Agent name", "coder")
        if agent_name == "\x00": info("Cancelled."); return True
        aa = _ask("Auto-approve shell commands? [Y/n]", "Y")
        if aa == "\x00": info("Cancelled."); return True
        auto_approve = aa.strip().lower() not in ("n", "no")
        p = Path(task_input)
        if p.exists() and p.suffix == ".md":
            agent_args = f"--tasks-file {shlex.quote(task_input)}"
        else:
            agent_args = f"--task {shlex.quote(task_input)}"

    elif template_name == "__custom__":
        tpl_path = _ask("Template file path (.md)", "")
        if tpl_path in ("\x00", ""):
            err("Template path is required.") if tpl_path != "\x00" else info("Cancelled.")
            return True
        template_name = tpl_path
        agent_name_default = Path(tpl_path).stem
        agent_name = _ask("Agent name", agent_name_default)
        if agent_name == "\x00": info("Cancelled."); return True
        extra = _ask("Extra args (optional)", "")
        if extra == "\x00": info("Cancelled."); return True
        aa = _ask("Auto-approve operations? [Y/n]", "Y")
        if aa == "\x00": info("Cancelled."); return True
        auto_approve = aa.strip().lower() not in ("n", "no")
        agent_args = extra

    # ── Interval ──────────────────────────────────────────────────────────
    interval_str = _ask("Seconds between iterations", "2")
    if interval_str == "\x00": info("Cancelled."); return True
    try:
        interval = max(0.5, float(interval_str))
    except ValueError:
        interval = 2.0

    # ── Confirm & start ───────────────────────────────────────────────────
    print()
    print(clr("  ─── Summary ───────────────────────────────────", "dim"))
    print(f"  Template  : {clr(template_name, 'cyan')}")
    print(f"  Name      : {clr(agent_name, 'white')}")
    print(f"  Args      : {agent_args or '(none)'}")
    print(f"  Interval  : {interval}s")
    print(f"  Auto-approve: {auto_approve}")
    print()

    confirm = _ask("Start? [Y/n]", "Y")
    if confirm in ("\x00",) or confirm.strip().lower() in ("n", "no"):
        info("Cancelled.")
        return True

    # Check for conflict
    existing = get_runner(agent_name)
    if existing:
        warn(f"Agent '{agent_name}' is already running ({existing.status}).")
        warn("Use /agent stop <name> first, or choose a different name.")
        return True

    send_fn = _get_bridge_send_fn(config)

    try:
        runner = start_runner(
            name=agent_name,
            template_name=template_name,
            args=agent_args,
            config=config,
            send_fn=send_fn,
            interval=interval,
            auto_approve=auto_approve,
        )
    except FileNotFoundError as e:
        err(str(e)); return True
    except Exception as e:
        err(f"Failed to start agent: {e}"); return True

    print()
    ok(f"Agent '{runner.name}' is running.")
    info(f"Log  : {runner._log_dir / 'log.jsonl'}")
    if send_fn:
        info("Progress → active bridge (Telegram / Slack / WeChat).")
    else:
        info("Progress → this terminal (iterations print here).")
    info(f"Stop : /agent stop {runner.name}")
    return True


# ── Main command ───────────────────────────────────────────────────────────

def cmd_agent(args: str, state, config) -> bool:
    """Autonomous agent loop — template-driven background agents.

    /agent                       — interactive wizard (recommended)
    /agent start <template> ...  — direct launch
    /agent stop <name|all>       — stop agent(s)
    /agent list                  — list running agents
    /agent status <name>         — recent iteration log
    /agent templates             — list available templates
    """
    from agent_runner import (
        list_templates, list_runners, start_runner, stop_runner,
        stop_all, get_runner,
    )

    parts = args.strip().split(None, 1) if args.strip() else []
    subcmd = parts[0].lower() if parts else ""
    rest   = parts[1] if len(parts) > 1 else ""

    # ── No args → wizard ──────────────────────────────────────────────────
    if not subcmd:
        return _wizard(config)

    # ── list ──────────────────────────────────────────────────────────────
    if subcmd in ("list", "ls"):
        runners = list_runners()
        if not runners:
            info("No agents running.")
            info("Start one with /agent  (wizard) or /agent start <template>")
            return True
        ok(f"{len(runners)} agent(s) running:")
        for r in runners:
            color = "green" if "running" in r.status else "yellow"
            print(f"  {clr('●', color)} {r.name:20s}  {r.status}")
            recs = r.recent_log(1)
            if recs:
                print(f"        {clr(recs[-1].summary[:80], 'dim')}")
        return True

    # ── templates ─────────────────────────────────────────────────────────
    if subcmd == "templates":
        templates = list_templates()
        if not templates:
            warn("No templates found.")
            return True
        ok(f"{len(templates)} template(s):")
        for t in templates:
            tag = clr(f"[{t['source']}]", "dim")
            print(f"  {clr(t['name'], 'cyan'):32s} {tag}")
        info("Use /agent to launch the wizard, or /agent start <name> [args]")
        return True

    # ── stop ──────────────────────────────────────────────────────────────
    if subcmd == "stop":
        target = rest.strip()
        if not target:
            err("Usage: /agent stop <name> | all")
            return True
        if target.lower() == "all":
            n = stop_all()
            ok(f"Stopped {n} agent(s).") if n else info("No agents running.")
        else:
            if stop_runner(target):
                ok(f"Agent '{target}' stopped.")
            else:
                warn(f"No running agent '{target}'.")
        return True

    # ── status / log ──────────────────────────────────────────────────────
    if subcmd in ("status", "log", "info"):
        name = rest.strip()
        if not name:
            err(f"Usage: /agent {subcmd} <name>")
            return True
        r = get_runner(name)
        if not r:
            warn(f"No running agent '{name}'.")
            return True
        print(r.summary_text())
        if subcmd == "log":
            info(f"Full log: {r._log_dir / 'log.jsonl'}")
        return True

    # ── start (power-user direct launch) ──────────────────────────────────
    if subcmd == "start":
        if not rest.strip():
            err("Usage: /agent start <template> [--name <n>] [--interval <s>] [args]")
            info("Available: " + ", ".join(t["name"] for t in list_templates()))
            info("Or just type /agent for the guided wizard.")
            return True

        try:
            tokens = shlex.split(rest)
        except ValueError as e:
            err(f"Parse error: {e}"); return True

        template_name   = tokens[0]
        remaining       = tokens[1:]
        agent_name      = Path(template_name).stem
        interval        = 2.0
        auto_approve    = True
        agent_args_parts: list[str] = []

        i = 0
        while i < len(remaining):
            tok = remaining[i]
            if tok == "--name" and i + 1 < len(remaining):
                agent_name = remaining[i + 1]; i += 2
            elif tok == "--interval" and i + 1 < len(remaining):
                try:
                    interval = float(remaining[i + 1])
                except ValueError:
                    err(f"--interval must be a number"); return True
                i += 2
            elif tok == "--no-auto-approve":
                auto_approve = False; i += 1
            else:
                agent_args_parts.append(tok); i += 1

        agent_args = " ".join(agent_args_parts)

        existing = get_runner(agent_name)
        if existing:
            warn(f"Agent '{agent_name}' already running ({existing.status}).")
            warn("Stop it first or choose --name <different>.")
            return True

        send_fn = _get_bridge_send_fn(config)

        try:
            runner = start_runner(
                name=agent_name,
                template_name=template_name,
                args=agent_args,
                config=config,
                send_fn=send_fn,
                interval=interval,
                auto_approve=auto_approve,
            )
        except FileNotFoundError as e:
            err(str(e)); return True
        except Exception as e:
            err(f"Failed to start: {e}"); return True

        ok(f"Agent '{runner.name}' started.")
        info(f"Template : {runner.template_path}")
        info(f"Args     : {agent_args or '(none)'}")
        info(f"Log      : {runner._log_dir / 'log.jsonl'}")
        if send_fn:
            info("Progress → bridge (Telegram / Slack / WeChat).")
        info(f"Stop with: /agent stop {runner.name}")
        return True

    # ── unknown ───────────────────────────────────────────────────────────
    err(f"Unknown subcommand '{subcmd}'.")
    info("Use /agent (wizard) · start · stop · list · status · templates")
    return True


# ── Bridge send_fn resolution ──────────────────────────────────────────────

def _get_bridge_send_fn(config: dict):
    """Return a send_fn that pushes to the first active bridge, or None."""
    import runtime
    ctx = runtime.get_session_ctx(config.get("_session_id", "default"))

    if ctx.tg_send:
        chat_id = config.get("telegram_chat_id", 0)
        token   = config.get("telegram_token", "")
        if chat_id and token:
            _s, _t, _c = ctx.tg_send, token, chat_id
            def _tg(text, s=_s, t=_t, c=_c):
                try: s(t, c, text)
                except Exception: pass
            return _tg

    if ctx.slack_send:
        channel = config.get("slack_channel", "")
        if channel:
            _s, _ch = ctx.slack_send, channel
            def _sl(text, s=_s, ch=_ch):
                try: s(ch, text)
                except Exception: pass
            return _sl

    if ctx.wx_send:
        uid = config.get("_wx_last_user", "")
        if uid:
            _s, _u = ctx.wx_send, uid
            def _wx(text, s=_s, u=_u):
                try: s(u, text)
                except Exception: pass
            return _wx

    return None
