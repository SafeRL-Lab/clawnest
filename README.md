

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/nano-claude-code)">
    <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/logo-v1.png" alt="Logo" width="280"> 
  </a>

  
<h1 align="center" style="font-size: 30px;"><strong><em>Nano Claude Code</em></strong>: A Minimal Python Reimplementation</h1>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">The newest source of Claude Code</a>
    ·
    <a href="https://github.com/SafeRL-Lab/nano-claude-code/issues">Issue</a>
  </p>
</div>

 <div align=center>
 <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>

---

## 🔥🔥🔥 News (Pacific Time)
- 12:20 PM, Apr 02, 2026: **v3.0** — Multi-agent packages (`multi_agent/`), memory package (`memory/`), skill package (`skill/`) with built-in skills, argument substitution, fork/inline execution, AI memory search, git worktree isolation, agent type definitions (**~5000** lines of Python)
- 10:00 AM, Apr 02, 2026: **v2.0** — Context compression, memory, sub-agents, skills, diff view, tool plugin system (**~3400** lines of Python Code), see [update](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/Update_README.MD)
- 01:47 PM, Apr 01, 2026: Support VLLM inference (**~2000** lines of Python Code)
- 11:30 AM, Apr 01, 2026: Support more **closed-source** models and **open-source models**: Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, and local open-source models via Ollama or any OpenAI-compatible endpoint. (**~1700** lines of Python Code)
- 09:50 AM, Apr 01, 2026: Support more **closed-source** models: Claude, GPT, Gemini. (**~1300** lines of Python Code)
- 08:23 AM, Apr 01, 2026: Release the initial version of Nano Claude Code (**~900 lines** of Python Code)

---

# Nano Claude Code

A minimal Python implementation of Claude Code in ~900 lines (Initial version), **supporting Claude, GPT, Gemini, Kimi, Qwen, Zhipu, DeepSeek, and local open-source models via Ollama or any OpenAI-compatible endpoint.**

---

## Content
  * [Features](#features)
  * [Supported Models](#supported-models)
  * [Installation](#installation)
  * [Usage: Closed-Source API Models](#usage--closed-source-api-models)
  * [Usage: Open-Source Models (Local)](#usage--open-source-models--local-)
  * [Model Name Format](#model-name-format)
  * [CLI Reference](#cli-reference)
  * [Slash Commands (REPL)](#slash-commands--repl-)
  * [Configuring API Keys](#configuring-api-keys)
  * [Permission System](#permission-system)
  * [Built-in Tools](#built-in-tools)
  * [Memory](#memory)
  * [Skills](#skills)
  * [Sub-Agents](#sub-agents)
  * [Context Compression](#context-compression)
  * [Diff View](#diff-view)
  * [CLAUDE.md Support](#claudemd-support)
  * [Session Management](#session-management)
  * [Project Structure](#project-structure)
  * [FAQ](#faq)




## Features

| Feature | Details |
|---|---|
| Multi-provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · Ollama · LM Studio · Custom endpoint |
| Interactive REPL | readline history, Tab-complete slash commands |
| Agent loop | Streaming API + automatic tool-use loop |
| 18 built-in tools | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList |
| Diff view | Git-style red/green diff display for Edit and Write |
| Context compression | Auto-compact long conversations to stay within model limits |
| Persistent memory | Dual-scope memory (user + project) with 4 types, AI search, staleness warnings |
| Multi-agent | Spawn typed sub-agents (coder/reviewer/researcher/…), git worktree isolation, background mode |
| Skills | Built-in `/commit` · `/review` + custom markdown skills with argument substitution and fork/inline execution |
| Plugin tools | Register custom tools via `tool_registry.py` |
| Permission system | `auto` / `accept-all` / `manual` modes |
| 17 slash commands | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · … |
| Context injection | Auto-loads `CLAUDE.md`, git status, cwd, persistent memory |
| Session persistence | Save / load conversations to `~/.nano_claude/sessions/` |
| Extended Thinking | Toggle on/off (Claude models only) |
| Cost tracking | Token usage + estimated USD cost |
| Non-interactive mode | `--print` flag for scripting / CI |

---

## Supported Models

### Closed-Source (API)

| Provider | Model | Context | Strengths | API Key Env |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | Most capable, best for complex reasoning | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | Balanced speed & quality | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | Fast, cost-efficient | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | Strong multimodal & coding | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | Fast, cheap | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | Strong reasoning | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | Advanced reasoning | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | Long context, multimodal | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | Fast, large context | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | Largest context window | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | Chinese & English | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | Chinese & English | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | Long context | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | Best Qwen quality | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | Balanced | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | Fast, cheap | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | Strong reasoning | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | Best GLM quality | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | General purpose | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | Free tier available | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | Strong coding | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | Chain-of-thought reasoning | `DEEPSEEK_API_KEY` |

### Open-Source (Local via Ollama)

| Model | Size | Strengths | Pull Command |
|---|---|---|---|
| `llama3.3` | 70B | General purpose, strong reasoning | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | Lightweight | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **Best for coding tasks** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | Chinese & English | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | Reasoning, math | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | Coding | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | Fast, efficient | `ollama pull mistral` |
| `mixtral` | 8x7B | Strong MoE model | `ollama pull mixtral` |
| `phi4` | 14B | Microsoft, strong reasoning | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Google open model | `ollama pull gemma3` |
| `codellama` | 7B / 34B | Code generation | `ollama pull codellama` |

> **Note:** Tool calling requires a model that supports function calling. Recommended local models: `qwen2.5-coder`, `llama3.3`, `mistral`, `phi4`.

---

## Installation

```bash
git clone <repo-url>
cd nano_claude_code

pip install -r requirements.txt
# or manually:
pip install anthropic openai httpx rich
```

---

## Usage: Closed-Source API Models

### Anthropic Claude

Get your API key at [console.anthropic.com](https://console.anthropic.com).

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# Default model (claude-opus-4-6)
python nano_claude.py

# Choose a specific model
python nano_claude.py --model claude-sonnet-4-6
python nano_claude.py --model claude-haiku-4-5-20251001

# Enable Extended Thinking
python nano_claude.py --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

Get your API key at [platform.openai.com](https://platform.openai.com).

```bash
export OPENAI_API_KEY=sk-...

python nano_claude.py --model gpt-4o
python nano_claude.py --model gpt-4o-mini
python nano_claude.py --model gpt-4.1-mini
python nano_claude.py --model o3-mini
```

### Google Gemini

Get your API key at [aistudio.google.com](https://aistudio.google.com).

```bash
export GEMINI_API_KEY=AIza...

python nano_claude.py --model gemini/gemini-2.0-flash
python nano_claude.py --model gemini/gemini-1.5-pro
python nano_claude.py --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi (Moonshot AI)

Get your API key at [platform.moonshot.cn](https://platform.moonshot.cn).

```bash
export MOONSHOT_API_KEY=sk-...

python nano_claude.py --model kimi/moonshot-v1-32k
python nano_claude.py --model kimi/moonshot-v1-128k
```

### Qwen (Alibaba DashScope)

Get your API key at [dashscope.aliyun.com](https://dashscope.aliyun.com).

```bash
export DASHSCOPE_API_KEY=sk-...

python nano_claude.py --model qwen/qwen-max
python nano_claude.py --model qwen/qwq-32b
python nano_claude.py --model qwen/qwen2.5-coder-32b-instruct
```

### Zhipu GLM

Get your API key at [open.bigmodel.cn](https://open.bigmodel.cn).

```bash
export ZHIPU_API_KEY=...

python nano_claude.py --model zhipu/glm-4-plus
python nano_claude.py --model zhipu/glm-4-flash   # free tier
```

### DeepSeek

Get your API key at [platform.deepseek.com](https://platform.deepseek.com).

```bash
export DEEPSEEK_API_KEY=sk-...

python nano_claude.py --model deepseek/deepseek-chat
python nano_claude.py --model deepseek/deepseek-reasoner
```

---

## Usage: Open-Source Models (Local)

### Option A — Ollama (Recommended)

Ollama runs models locally with zero configuration. No API key required.

**Step 1: Install Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

**Step 2: Pull a model**

```bash
# Best for coding (recommended)
ollama pull qwen2.5-coder          # 4.7 GB (7B)
ollama pull qwen2.5-coder:32b      # 19 GB (32B)

# General purpose
ollama pull llama3.3               # 42 GB (70B)
ollama pull llama3.2               # 2.0 GB (3B)

# Reasoning
ollama pull deepseek-r1            # 4.7 GB (7B)
ollama pull deepseek-r1:32b        # 19 GB (32B)

# Other
ollama pull phi4                   # 9.1 GB (14B)
ollama pull mistral                # 4.1 GB (7B)
```

**Step 3: Start Ollama server** (runs automatically on macOS; on Linux run manually)

```bash
ollama serve     # starts on http://localhost:11434
```

**Step 4: Run nano claude**

```bash
python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model ollama/llama3.3
python nano_claude.py --model ollama/deepseek-r1
```

**List your locally available models:**

```bash
ollama list
```

Then use any model from the list:

```bash
python nano_claude.py --model ollama/<model-name>
```

---

### Option B — LM Studio

LM Studio provides a GUI to download and run models, with a built-in OpenAI-compatible server.

**Step 1:** Download [LM Studio](https://lmstudio.ai) and install it.

**Step 2:** Search and download a model inside LM Studio (GGUF format).

**Step 3:** Go to **Local Server** tab → click **Start Server** (default port: 1234).

**Step 4:**

```bash
python nano_claude.py --model lmstudio/<model-name>
# e.g.:
python nano_claude.py --model lmstudio/phi-4-GGUF
python nano_claude.py --model lmstudio/qwen2.5-coder-7b
```

The model name should match what LM Studio shows in the server status bar.

---

### Option C — vLLM / Self-Hosted OpenAI-Compatible Server

For self-hosted inference servers (vLLM, TGI, llama.cpp server, etc.) that expose an OpenAI-compatible API:

Quick Start for option C:
Step 1: Start vllm:
 ```
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
```


 Step 2: Start nano claude：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  python nano_claude.py --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# Example: vLLM serving Qwen2.5-Coder-32B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000

# Then run nano claude pointing to your server:
python nano_claude.py
```

Inside the REPL:

```
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123    # skip if no auth
/model custom/Qwen2.5-Coder-32B-Instruct
```

Or set via environment:

```bash
export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123

python nano_claude.py --model custom/Qwen2.5-Coder-32B-Instruct
```

For a remote GPU server:

```bash
/config custom_base_url=http://192.168.1.100:8000/v1
/model custom/your-model-name
```

---

## Model Name Format

Three equivalent formats are supported:

```bash
# 1. Auto-detect by prefix (works for well-known models)
python nano_claude.py --model gpt-4o
python nano_claude.py --model gemini-2.0-flash
python nano_claude.py --model deepseek-chat

# 2. Explicit provider prefix with slash
python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model kimi/moonshot-v1-128k

# 3. Explicit provider prefix with colon (also works)
python nano_claude.py --model kimi:moonshot-v1-32k
python nano_claude.py --model qwen:qwen-max
```

**Auto-detection rules:**

| Model prefix | Detected provider |
|---|---|
| `claude-` | anthropic |
| `gpt-`, `o1`, `o3` | openai |
| `gemini-` | gemini |
| `moonshot-`, `kimi-` | kimi |
| `qwen`, `qwq-` | qwen |
| `glm-` | zhipu |
| `deepseek-` | deepseek |
| `llama`, `mistral`, `phi`, `gemma`, `mixtral`, `codellama` | ollama |

---

## CLI Reference

```
python nano_claude.py [OPTIONS] [PROMPT]

Options:
  -p, --print          Non-interactive: run prompt and exit
  -m, --model MODEL    Override model (e.g. gpt-4o, ollama/llama3.3)
  --accept-all         Auto-approve all operations (no permission prompts)
  --verbose            Show thinking blocks and per-turn token counts
  --thinking           Enable Extended Thinking (Claude only)
  --version            Print version and exit
  -h, --help           Show help
```

**Examples:**

```bash
# Interactive REPL with default model
python nano_claude.py

# Switch model at startup
python nano_claude.py --model gpt-4o
python nano_claude.py -m ollama/deepseek-r1:32b

# Non-interactive / scripting
python nano_claude.py --print "Write a Python fibonacci function"
python nano_claude.py -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / automation (no permission prompts)
python nano_claude.py --accept-all --print "Initialize a Python project with pyproject.toml"

# Debug mode (see tokens + thinking)
python nano_claude.py --thinking --verbose
```

---

## Slash Commands (REPL)

Type `/` and press **Tab** to autocomplete.

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear conversation history |
| `/model` | Show current model + list all available models |
| `/model <name>` | Switch model (takes effect immediately) |
| `/config` | Show all current config values |
| `/config key=value` | Set a config value (persisted to disk) |
| `/save` | Save session (auto-named by timestamp) |
| `/save <filename>` | Save session to named file |
| `/load` | List all saved sessions |
| `/load <filename>` | Load a saved session |
| `/history` | Print full conversation history |
| `/context` | Show message count and token estimate |
| `/cost` | Show token usage and estimated USD cost |
| `/verbose` | Toggle verbose mode (tokens + thinking) |
| `/thinking` | Toggle Extended Thinking (Claude only) |
| `/permissions` | Show current permission mode |
| `/permissions <mode>` | Set permission mode: `auto` / `accept-all` / `manual` |
| `/cwd` | Show current working directory |
| `/cwd <path>` | Change working directory |
| `/memory` | List all persistent memories |
| `/memory <query>` | Search memories by keyword |
| `/skills` | List available skills |
| `/agents` | Show sub-agent task status |
| `/exit` / `/quit` | Exit |

**Switching models inside a session:**

```
[myproject] ❯ /model
  Current model: claude-opus-4-6  (provider: anthropic)

  Available models by provider:
    anthropic     claude-opus-4-6, claude-sonnet-4-6, ...
    openai        gpt-4o, gpt-4o-mini, o3-mini, ...
    ollama        llama3.3, llama3.2, phi4, mistral, ...
    ...

[myproject] ❯ /model gpt-4o
  Model set to gpt-4o  (provider: openai)

[myproject] ❯ /model ollama/qwen2.5-coder
  Model set to ollama/qwen2.5-coder  (provider: ollama)
```

---

## Configuring API Keys

### Method 1: Environment Variables (recommended)

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...       # Kimi
export DASHSCOPE_API_KEY=sk-...      # Qwen
export ZHIPU_API_KEY=...             # Zhipu GLM
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
```

### Method 2: Set Inside the REPL (persisted)

```
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
```

Keys are saved to `~/.nano_claude/config.json` and loaded automatically on next launch.

### Method 3: Edit the Config File Directly

```json
// ~/.nano_claude/config.json
{
  "model": "qwen/qwen-max",
  "max_tokens": 8192,
  "permission_mode": "auto",
  "verbose": false,
  "thinking": false,
  "qwen_api_key": "sk-...",
  "kimi_api_key": "sk-...",
  "deepseek_api_key": "sk-..."
}
```

---

## Permission System

| Mode | Behavior |
|---|---|
| `auto` (default) | Read-only operations always allowed. Prompts before Bash commands and file writes. |
| `accept-all` | Never prompts. All operations proceed automatically. |
| `manual` | Prompts before every single operation, including reads. |

**When prompted:**

```
  Allow: Run: git commit -am "fix bug"  [y/N/a(ccept-all)]
```

- `y` — approve this one action
- `n` or Enter — deny
- `a` — approve and switch to `accept-all` for the rest of the session

**Commands always auto-approved in `auto` mode:**
`ls`, `cat`, `head`, `tail`, `wc`, `pwd`, `echo`, `git status`, `git log`, `git diff`, `git show`, `find`, `grep`, `rg`, `python`, `node`, `pip show`, `npm list`, and other read-only shell commands.

---

## Built-in Tools

### Core Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Read` | Read file with line numbers | `file_path`, `limit`, `offset` |
| `Write` | Create or overwrite file (shows diff) | `file_path`, `content` |
| `Edit` | Exact string replacement (shows diff) | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | Execute shell command | `command`, `timeout` (default 30s) |
| `Glob` | Find files by glob pattern | `pattern` (e.g. `**/*.py`), `path` |
| `Grep` | Regex search in files (uses ripgrep if available) | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | Fetch and extract text from URL | `url`, `prompt` |
| `WebSearch` | Search the web via DuckDuckGo | `query` |

### Memory Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `MemorySave` | Save or update a persistent memory | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | Delete a memory by name | `name`, `scope` |
| `MemorySearch` | Search memories by keyword (or AI ranking) | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | List all memories with age and metadata | `scope` |

### Sub-Agent Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Agent` | Spawn a sub-agent for a task | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | Send a message to a named background agent | `name`, `message` |
| `CheckAgentResult` | Check status/result of a background agent | `task_id` |
| `ListAgentTasks` | List all active and finished agent tasks | — |
| `ListAgentTypes` | List available agent type definitions | — |

### Skill Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `Skill` | Invoke a skill by name from within the conversation | `name`, `args` |
| `SkillList` | List all available skills with triggers and metadata | — |

> **Adding custom tools:** See [Architecture Guide](docs/architecture.md#tool-registry) for how to register your own tools.

---

## Memory

The model can remember things across conversations using the built-in memory system.

**How it works:** Memories are stored as markdown files. There are two scopes:
- **User scope** (`~/.nano_claude/memory/`) — follows you across all projects
- **Project scope** (`.nano_claude/memory/` in cwd) — specific to the current repo

A `MEMORY.md` index (≤ 200 lines / 25 KB) is auto-rebuilt on every save or delete and injected into the system prompt so Claude always has an overview.

**Memory types:**

| Type | Use for |
|---|---|
| `user` | Your role, preferences, background |
| `feedback` | How you want the model to behave |
| `project` | Ongoing work, deadlines, decisions |
| `reference` | Links to external resources |

**Memory file format** (`~/.nano_claude/memory/coding_style.md`):
```markdown
---
name: coding style
description: Python formatting preferences
type: feedback
created: 2026-04-02
---
Prefer 4-space indentation and full type hints in all Python code.
**Why:** user explicitly stated this preference.
**How to apply:** apply to every Python file written or edited.
```

**Example interaction:**

```
You: Remember that I prefer 4-space indentation and type hints in all Python code.
AI: [calls MemorySave] Memory saved: coding_style [feedback/user]

You: /memory
  [feedback/user] coding_style (today): Python formatting preferences

You: /memory python
  [feedback/user] coding_style: Prefers 4-space indent and type hints in Python
```

**Staleness warnings:** Memories older than 1 day get a freshness note in `/memory` output so you know when to review or update them.

**AI-ranked search:** `MemorySearch(query="...", use_ai=true)` uses the model to rank results by relevance rather than simple keyword matching.

---

## Skills

Skills are reusable prompt templates that give the model specialized capabilities. Two built-in skills ship out of the box — no setup required.

**Built-in skills:**

| Trigger | Description |
|---|---|
| `/commit` | Review staged changes and create a well-structured git commit |
| `/review [PR]` | Review code or PR diff with structured feedback |

**Quick start — custom skill:**

```bash
mkdir -p ~/.nano_claude/skills
```

Create `~/.nano_claude/skills/deploy.md`:

```markdown
---
name: deploy
description: Deploy to an environment
triggers: [/deploy]
allowed-tools: [Bash, Read]
when_to_use: Use when the user wants to deploy a version to an environment.
argument-hint: [env] [version]
arguments: [env, version]
context: inline
---

Deploy $VERSION to the $ENV environment.
Full args: $ARGUMENTS
```

Now use it:

```
You: /deploy staging 2.1.0
AI: [deploys version 2.1.0 to staging]
```

**Argument substitution:**
- `$ARGUMENTS` — the full raw argument string
- `$ARG_NAME` — positional substitution by named argument (first word → first name)
- Missing args become empty strings

**Execution modes:**
- `context: inline` (default) — runs inside current conversation history
- `context: fork` — runs as an isolated sub-agent with fresh history; supports `model` override

**Priority** (highest wins): project-level > user-level > built-in

**List skills:** `/skills` — shows triggers, argument hint, source, and `when_to_use`

**Skill search paths:**

```
./.nano_claude/skills/     # project-level (overrides user-level)
~/.nano_claude/skills/     # user-level
```

---

## Sub-Agents

The model can spawn independent sub-agents to handle tasks in parallel.

**Specialized agent types** — built-in:

| Type | Optimized for |
|---|---|
| `general-purpose` | Research, exploration, multi-step tasks |
| `coder` | Writing, reading, and modifying code |
| `reviewer` | Security, correctness, and code quality analysis |
| `researcher` | Web search and documentation lookup |
| `tester` | Writing and running tests |

**Basic usage:**
```
You: Search this codebase for all TODO comments and summarize them.
AI: [calls Agent(prompt="...", subagent_type="researcher")]
    Sub-agent reads files, greps for TODOs...
    Result: Found 12 TODOs across 5 files...
```

**Background mode** — spawn without waiting, collect result later:
```
AI: [calls Agent(prompt="run all tests", name="test-runner", wait=false)]
AI: [continues other work...]
AI: [calls CheckAgentResult / SendMessage to follow up]
```

**Git worktree isolation** — agents work on an isolated branch with no conflicts:
```
Agent(prompt="refactor auth module", isolation="worktree")
```
The worktree is auto-cleaned up if no changes were made; otherwise the branch name is reported.

**Custom agent types** — create `~/.nano_claude/agents/myagent.md`:
```markdown
---
name: myagent
description: Specialized for X
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
Extra system prompt for this agent type.
```

**List running agents:** `/agents`

Sub-agents have independent conversation history, share the file system, and are limited to 3 levels of nesting.

---

## Context Compression

Long conversations are automatically compressed to stay within the model's context window.

**Two layers:**

1. **Snip** — Old tool outputs (file reads, bash results) are truncated after a few turns. Fast, no API cost.
2. **Auto-compact** — When token usage exceeds 70% of the context limit, older messages are summarized by the model into a concise recap.

This happens transparently. You don't need to do anything.

---

## Diff View

When the model edits or overwrites a file, you see a git-style diff:

```diff
  Changes applied to config.py:

--- a/config.py
+++ b/config.py
@@ -12,7 +12,7 @@
     "model": "claude-opus-4-6",
-    "max_tokens": 8192,
+    "max_tokens": 16384,
     "permission_mode": "auto",
```

Green lines = added, red lines = removed. New file creations show a summary instead.

---

## CLAUDE.md Support

Place a `CLAUDE.md` file in your project to give the model persistent context about your codebase. Nano Claude automatically finds and injects it into the system prompt.

```
~/.claude/CLAUDE.md          # Global — applies to all projects
/your/project/CLAUDE.md      # Project-level — found by walking up from cwd
```

**Example `CLAUDE.md`:**

```markdown
# Project: FastAPI Backend

## Stack
- Python 3.12, FastAPI, PostgreSQL, SQLAlchemy 2.0, Alembic
- Tests: pytest, coverage target 90%

## Conventions
- Format with black, lint with ruff
- Full type annotations required
- New endpoints must have corresponding tests

## Important Notes
- Never hard-code credentials — use environment variables
- Do not modify existing Alembic migration files
- The `staging` branch deploys automatically to staging on push
```

---

## Session Management

```bash
# Inside REPL:
/save                          # auto-name: session_20260401_143022.json
/save debug_auth_bug           # named save

/load                          # list all saved sessions
/load debug_auth_bug           # resume a session
/load session_20260401_143022.json
```

Sessions are stored as JSON in `~/.nano_claude/sessions/`.

---

## Project Structure

```
nano_claude_code/
├── nano_claude.py        # Entry point: REPL + slash commands + diff rendering
├── agent.py              # Agent loop: streaming, tool dispatch, compaction
├── providers.py          # Multi-provider: Anthropic, OpenAI-compat streaming
├── tools.py              # Core tools (Read/Write/Edit/Bash/Glob/Grep/Web) + registry wiring
├── tool_registry.py      # Tool plugin registry: register, lookup, execute
├── compaction.py         # Context compression: snip + auto-summarize
├── context.py            # System prompt builder: CLAUDE.md + git + memory
├── config.py             # Config load/save/defaults
│
├── multi_agent/          # Multi-agent package
│   ├── __init__.py       # Re-exports
│   ├── subagent.py       # AgentDefinition, SubAgentManager, worktree helpers
│   └── tools.py          # Agent, SendMessage, CheckAgentResult, ListAgentTasks, ListAgentTypes
├── subagent.py           # Backward-compat shim → multi_agent/
│
├── memory/               # Memory package
│   ├── __init__.py       # Re-exports
│   ├── types.py          # MEMORY_TYPES and format guidance
│   ├── store.py          # save/load/delete/search, MEMORY.md index rebuilding
│   ├── scan.py           # MemoryHeader, age/freshness helpers
│   ├── context.py        # get_memory_context(), truncation, AI search
│   └── tools.py          # MemorySave, MemoryDelete, MemorySearch, MemoryList
├── memory.py             # Backward-compat shim → memory/
│
├── skill/                # Skill package
│   ├── __init__.py       # Re-exports; imports builtin to register built-ins
│   ├── loader.py         # SkillDef, parse, load_skills, find_skill, substitute_arguments
│   ├── builtin.py        # Built-in skills: /commit, /review
│   ├── executor.py       # execute_skill(): inline or forked sub-agent
│   └── tools.py          # Skill, SkillList
├── skills.py             # Backward-compat shim → skill/
│
└── tests/                # 101 unit tests
    ├── test_memory.py
    ├── test_skills.py
    ├── test_subagent.py
    ├── test_tool_registry.py
    ├── test_compaction.py
    └── test_diff_view.py
```

> **For developers:** Each feature package (`multi_agent/`, `memory/`, `skill/`) is self-contained. Add custom tools by calling `register_tool(ToolDef(...))` from any module imported by `tools.py`.

---

## FAQ

**Q: Tool calls don't work with my local Ollama model.**

Not all models support function calling. Use one of the recommended tool-calling models: `qwen2.5-coder`, `llama3.3`, `mistral`, or `phi4`.

```bash
ollama pull qwen2.5-coder
python nano_claude.py --model ollama/qwen2.5-coder
```

**Q: How do I connect to a remote GPU server running vLLM?**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q: How do I check my API cost?**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Est. cost:     $0.0648 USD
```

**Q: Can I use multiple API keys in the same session?**

Yes. Set all the keys you need upfront (via env vars or `/config`). Then switch models freely — each call uses the key for the active provider.

**Q: How do I make a model available across all projects?**

Add keys to `~/.bashrc` or `~/.zshrc`. Set the default model in `~/.nano_claude/config.json`:

```json
{ "model": "claude-sonnet-4-6" }
```

**Q: Qwen / Zhipu returns garbled text.**

Ensure your `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` is correct and the account has sufficient quota. Both providers use UTF-8 and handle Chinese well.

**Q: Can I pipe input to nano claude?**

```bash
echo "Explain this file" | python nano_claude.py --print --accept-all
cat error.log | python nano_claude.py -p "What is causing this error?"
```

**Q: How do I run it as a CLI tool from anywhere?**

```bash
# Add an alias to ~/.bashrc or ~/.zshrc
alias nc='python /path/to/nano_claude_code/nano_claude.py'

# Or install as a script
pip install -e .   # if setup.py exists
```
