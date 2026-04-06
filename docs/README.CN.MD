中文 | [English](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/README.md) | [Français](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.FR.MD) | [한국어](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.KO.MD) | [日本語](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.JP.MD) | [Deutsch](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.DE.MD) | [Português](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/README.ES.MD)

<div align="center">
  <a href="[https://github.com/SafeRL-Lab/Robust-Gymnasium](https://github.com/SafeRL-Lab/nano-claude-code)">
    <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/logo-v1.png" alt="Logo" width="280"> 
  </a>


<h1 align="center" style="font-size: 30px;"><strong><em>Nano Claude Code</em></strong>：一个快速、易用、支持任意模型的 Claude Code Python 重实现</h1>
<p align="center">
    <a href="https://github.com/chauncygu/collection-claude-code-source-code">Claude Code 最新源码合集</a>
    ·
    <a href="https://github.com/SafeRL-Lab/nano-claude-code/issues">问题反馈</a>
  ·
    <a href="https://deepwiki.com/SafeRL-Lab/nano-claude-code">简要介绍</a>

  </p>
</div>

 <div align=center>
 <img src="https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/demo.gif" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> </center>
 </div>

---

## 🔥🔥🔥 更新日志（太平洋时间）

- 2026 年 4 月 5 日（**v3.05.5**）：**交互式 Ollama 模型选择器、Windows 修复、/brainstorm 命令、Rich Live SSH 修复**
  - **交互式 Ollama 模型选择器** —— 当请求因 404（模型不存在）失败时，nano-claude 会查询本地 Ollama API（`/api/tags`），并展示一个带编号的模型选择器，以便无需重启即可切换模型并重试。取消操作时会优雅退出，不会导致 REPL 崩溃。
  - **Windows 文件处理** —— `tools.py` 中的 `_read`、`_write` 和 `_edit` 现在强制使用 UTF-8 编码与 `newline=""`。`_edit` 会检测纯 CRLF 文件（每个 `\n` 都属于 `\r\n`），并在编辑后恢复换行格式；混合换行文件则保持原样，避免损坏。
  - **/brainstorm 命令** —— `/brainstorm [topic]` 可运行一个多人格 AI 辩论。模型会先根据主题生成 N 个专家人格（如地缘政治 → 分析师与外交官；软件 → 架构师与工程师等）。代理数量在运行时交互式选择（2–100，默认 5）。结果保存到 `brainstorm_outputs/`，并由主代理进行综合。
  - **Rich Live SSH 修复** —— 在 SSH 会话中（检测到 `SSH_CLIENT`/`SSH_TTY`）会自动禁用 Rich 的原位 Live 流式渲染，因为 ANSI 光标上移在 SSH 中可能失效并造成重复输出。可通过 `/config rich_live=true/false` 覆盖。
  - **`threading.RLock`** —— 将 `threading.Lock` 替换为 `RLock`，以支持 brainstorm 综合过程和 Ollama 重试路径中的可重入调用。

- 2026 年 4 月 5 日 05:39 PM（**v3.05.4**）：**推理、渲染与打包改进，增强记忆系统，本地 Ollama 模型原生视觉支持，Bracketed Paste Mode，Rich Tab 补全**
  - **Bracketed Paste Mode** —— 用标准终端 Bracketed Paste Mode 协议替换了旧的基于时间的多行粘贴检测。任意长度的粘贴文本（代码块、长提示词、多段说明）现在都会被作为一个完整 turn 收集，零延迟且不会产生空行伪影。对于不支持 BPM 的终端，会回退到 60 ms 的时间窗口。REPL 退出时会正确关闭 bracketed paste mode。
  - **带描述的 Rich Tab 补全** —— 在输入 `/` 后按 Tab，现在会显示所有命令、对应的一行描述，以及其子命令提示。输入 `/plugin ` 后按 Tab，会列出所有子命令（`install`、`uninstall`、`enable` 等）。若仅有一个命令匹配当前前缀，则会自动补全。支持 `/mcp`、`/plugin`、`/tasks`、`/cloudsave`、`/voice`、`/permissions`、`/proactive` 和 `/memory` 的子命令补全。
  - **模型名 Bug 修复** —— `--model ollama/qwen3.5:35b` 不会再被错误转换成 `ollama/qwen3.5/35b`。启动时的冒号转斜杠逻辑现在只会在 `:` 左侧是已知 provider 且模型名中尚无 `/` 时触发，从而保留 Ollama 的 `model:tag` 格式。
  - **本地 Ollama 模型原生视觉支持**（`llava`、`gemma4`、`llama3.2-vision`）：新增 `/image [prompt]` 命令，可抓取当前剪贴板图片、编码为 Base64，并附加到下一次提示中。使用 `pip install nano-claude-code[vision]` 安装 Pillow；Linux 用户还需要 `xclip`（`sudo apt install xclip`）。
  - **增强记忆系统** —— 为每条记忆加入 `confidence` / `source` / `last_used_at` / `conflict_group` 元数据；`MemorySave` 在覆盖前进行冲突检测并提醒；`MemorySearch` 通过 `confidence × recency`（30 天衰减）重新排序结果，并在命中时更新 `last_used_at`；新增 `/memory consolidate` 命令，会对当前会话做轻量 AI 分析，并自动保存最多 3 条长期洞见（用户偏好、反馈修正、项目决策），置信度为 0.8 —— 永远不会覆盖更高置信度的用户记忆。
  - **合并后修复** —— 删除了每次 OpenAI-compatible API 调用时都会触发的调试文件写入 `debug_payload.json`（这是 PR #11 开发中遗留的问题）。同时修复了 thinking block 结束后 ANSI dim 颜色未复位的问题，这会导致后续文本在非 Rich 终端中变暗。`pyproject.toml` 版本号提升至 `3.05.4`，并将 `sounddevice` 移至可选 `voice` extra（`pip install nano-claude-code[voice]`）。
  - **原生 Ollama 推理 + 终端渲染修复** —— 本地推理模型（`deepseek-r1`、`qwen3`、`gemma4`）现在会将其 `<think>` 块流式输出到终端。Ollama 在 `msg["thinking"]` 中暴露思维内容，但 nano-claude 之前会丢弃它们；现在通过从 Ollama 适配器中产出 `ThinkingChunk` 事件进行修复。同时修复了 Windows CMD/PowerShell 中由于逐 token 的 ANSI dim 重置导致思维内容竖排输出的问题，并修正了 `flush_response()` 逻辑，使其只在结束时执行一次，而不是每个 thinking token 都执行。通过 `/verbose` 与 `/thinking` 启用。
  - **uv 支持** —— 新增 `pyproject.toml`；可使用 `uv tool install .` 安装，使 `nano_claude` 命令能在隔离环境中全局使用，无需手动配置 PATH。
- 2026 年 4 月 5 日 00:41 PM：**v3.05.3 增加结构化会话历史** —— 结构化会话历史：每次退出时，会话都会保存到 `daily/YYYY-MM-DD/`（上限由 `session_daily_limit` 控制，默认每天 5 个），并追加到总历史 `history.json`（上限由 `session_history_limit` 控制，默认 100）。每个会话文件现在还包含 `session_id` 和 `saved_at` 元数据。`/load` 会按日期对会话分组，并显示时间、ID 与 turn 数；支持多选（`1,2,3`）来合并会话，也支持用 `H` 加载完整历史，并在加载前确认 token 数估计。这两个上限都可通过 `/config` 配置。
- 2026 年 4 月 5 日 00:41 PM：**v3.05.3 fix session** —— 结构化会话历史：每次退出时，会话都会保存到 `daily/YYYY-MM-DD/`（上限由 `session_daily_limit` 控制，默认每天 5 个），并追加到总历史 `history.json`（上限由 `session_history_limit` 控制，默认 100）。每个会话文件现在还包含 `session_id` 和 `saved_at` 元数据。`/load` 会按日期对会话分组，并显示时间、ID 与 turn 数；支持多选（`1,2,3`）来合并会话，也支持用 `H` 加载完整历史，并在加载前确认 token 数估计。这两个上限都可通过 `/config` 配置。
- 2026 年 4 月 5 日 09:34 AM：**v3.05.3** —— 新增 GitHub Gist 云同步：`/cloudsave setup <token>` 用于配置，`/cloudsave` 将当前会话上传到私有 Gist，`/cloudsave auto on` 在 `/exit` 时自动同步，`/cloudsave list` 浏览云端会话，`/cloudsave load <id>` 从云端恢复。使用的是 Python 标准库 `urllib` —— 无需新增依赖。同时在启动横幅中显示版本号（例如 `v3.05.2`）：启动时会以绿色显示当前版本，便于一眼确认正在运行的版本。
- 2026 年 4 月 5 日 08:58 AM：**v3.05.2** —— 引入 `/proactive [duration]` 命令：后台守护线程会监控用户是否处于非活跃状态，并在指定时间间隔后自动唤醒代理（例如 `/proactive 5m`），从而无需用户干预即可实现持续监控循环。现在 `/proactive` 无参数时会显示当前状态；`/proactive off` 会显式关闭。主动轮询状态保存在 `config` 中（不使用模块级全局变量）。监视器异常会通过 `traceback` 记录，而不是静默吞掉。同时修复了启用 Rich 终端时的重复输出问题：通过在流式过程中缓冲文本，并使用 `rich.live.Live` 一次性渲染 Markdown，实现真正的原位流式 Markdown 显示。
- 2026 年 4 月 4 日 10:51 PM：**v3.05_fix04** —— 修复了 `/model` 和配置保存命令在新引入的 `_run_query_callback` 被序列化为 JSON 时导致崩溃的问题；同时还把 `SleepTimer` 的使用说明加入系统提示中，以便代理能更主动地调用后台计时器。
- 2026 年 4 月 4 日 10:28 PM：**v3.05_fix03** —— 新增原生 `SleepTimer` 工具，允许代理安排后台定时器，并在延迟结束后自主唤醒自己 —— 无需用户再次输入。配合 `threading.Lock`，避免后台与前台调用同时发生时输出冲突。还包括多项跨平台修复：Windows ANSI 颜色支持、支持 CRLF 的 Edit 工具匹配、为 `/load` 增加交互式编号菜单、通过 `/api/chat` 原生支持 Ollama 流式输出，以及根据 provider 自动限制 `max_tokens`，避免 API 报错。
- 2026 年 4 月 4 日 08:31 PM：**v3.05_fix** —— Autosave + `/resume`：会话会在 `/exit`、`/quit`、`Ctrl+C` 和 `Ctrl+D` 时自动保存到 `mr_sessions/session_latest.json`。使用 `/resume` 可立即恢复最近一次会话，或使用 `/resume <file>` 从 `mr_sessions/` 加载指定文件。同时增强了对 API 与本地 Ollama 模型（尤其是 gemma4）的支持，改进了 Windows 兼容性、会话管理体验，以及 Edit 工具的跨平台可靠性。
- 2026 年 4 月 4 日 00:41 AM：**v3.05** —— 语音输入（`voice/` 包）：录音后端为 `sounddevice` → `arecord` → SoX，STT 后端为 `faster-whisper` → `openai-whisper` → OpenAI API。会从 git 分支、项目名和最近文件中智能提取关键词，并作为 Whisper 的 `initial_prompt` 传入，以提升代码领域识别准确率。新增 REPL 命令 `/voice`、`/voice status`、`/voice lang <code>`。完全支持离线运行，无需 API Key。新增 29 个测试（约 **11.6K** 行 Python 代码）。
- 2026 年 4 月 3 日 10:29 PM：**v3.04** —— 扩展工具覆盖范围：`NotebookEdit`（编辑 Jupyter `.ipynb` 单元格 —— 支持替换/插入/删除，并进行完整 JSON round-trip）和 `GetDiagnostics`（通过 pyright/mypy/flake8/tsc/shellcheck 提供类 LSP 的诊断）。同时通过改用基于名称的查找，修复了 `_register_builtins` 中原有的 schema-index bug（约 **10.5K** 行 Python 代码）。
- 2026 年 4 月 3 日 06:00 PM：**v3.03** —— 任务管理系统（`task/` 包）：`TaskCreate` / `TaskUpdate` / `TaskGet` / `TaskList` 工具，支持顺序 ID、依赖边（blocks/blocked_by）、元数据、持久化到 `.nano_claude/tasks.json`、线程安全存储，以及 `/tasks` REPL 命令。新增 37 个测试（约 **9500** 行 Python 代码）。
- 2026 年 4 月 3 日 02:50 PM：**v3.02** —— 插件系统（`plugin/` 包）：通过 `/plugin` CLI 进行 install/uninstall/enable/disable/update；支持推荐引擎（关键词+标签匹配）、多作用域（user/project）、基于 git 的插件市场。新增 `AskUserQuestion` 工具：任务中途可以交互式地向用户提问，支持编号选项和自由文本输入（约 **~8500** 行 Python 代码）。
- 2026 年 4 月 3 日 10:00 AM：**v3.01** —— MCP（Model Context Protocol）支持：`mcp/` 包、stdio + SSE + HTTP 传输、自动工具发现、`/mcp` 命令、34 个新测试（约 **~7000** 行 Python 代码）。
- 2026 年 4 月 2 日 12:20 PM：**v3.0** —— 多代理包（`multi_agent/`）、记忆包（`memory/`）、技能包（`skill/`），包含内置技能、参数替换、fork/inline 执行、AI 记忆检索、git worktree 隔离、代理类型定义（约 **~5000** 行 Python 代码），详见 [update](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/update_readme_v3.0.md)。
- 2026 年 4 月 2 日 10:00 AM：**v2.0** —— 上下文压缩、记忆、子代理、技能、diff 视图、工具插件系统（约 **~3400** 行 Python 代码）。
- 2026 年 4 月 1 日 01:47 PM：支持 VLLM 推理（约 **~2000** 行 Python 代码）。
- 2026 年 4 月 1 日 11:30 AM：支持更多**闭源模型**和**开源模型**：Claude、GPT、Gemini、Kimi、Qwen、Zhipu、DeepSeek，以及通过 Ollama 或任何 OpenAI-compatible endpoint 使用的本地开源模型。（约 **~1700** 行 Python 代码）。
- 2026 年 4 月 1 日 09:50 AM：支持更多**闭源模型**：Claude、GPT、Gemini。（约 **~1300** 行 Python 代码）。
- 2026 年 4 月 1 日 08:23 AM：发布 Nano Claude Code 初始版本（约 **~900** 行 Python 代码）。

---

# Nano Claude Code

Nano Claude Code：**一个轻量级**、**易于使用**、**支持任意模型**的 Claude Code Python 重实现，例如 Claude、GPT、Gemini、Kimi、Qwen、Zhipu、DeepSeek，以及通过 Ollama 或任意 OpenAI-compatible endpoint 使用的本地开源模型。

---

## 目录
  * [为什么选择 Nano Claude Code](#为什么选择-nano-claude-code)
  * [特性](#特性)
  * [支持的模型](#支持的模型)
  * [安装](#安装)
  * [使用：闭源 API 模型](#使用闭源-api-模型)
  * [使用：开源模型（本地）](#使用开源模型本地)
  * [模型名称格式](#模型名称格式)
  * [CLI 参考](#cli-参考)
  * [Slash 命令（REPL）](#slash-命令repl)
  * [配置 API Keys](#配置-api-keys)
  * [权限系统](#权限系统)
  * [内置工具](#内置工具)
  * [记忆系统](#记忆系统)
  * [技能](#技能)
  * [子代理](#子代理)
  * [MCP（Model Context Protocol）](#mcpmodel-context-protocol)
  * [插件系统](#插件系统)
  * [AskUserQuestion 工具](#askuserquestion-工具)
  * [任务管理](#任务管理)
  * [语音输入](#语音输入)
  * [Brainstorm](#brainstorm)
  * [主动式后台监控](#主动式后台监控)
  * [上下文压缩](#上下文压缩)
  * [Diff 视图](#diff-视图)
  * [CLAUDE.md 支持](#claudemd-支持)
  * [会话管理](#会话管理)
  * [云同步（GitHub Gist）](#云同步github-gist)
  * [项目结构](#项目结构)
  * [FAQ](#faq)

## 为什么选择 Nano Claude Code

Claude Code 是一个强大、面向生产环境的 AI 编程助手 —— 但它的源码是一个编译后的 12 MB TypeScript/Node.js bundle（约 1,300 个文件、约 283K 行代码）。它与 Anthropic API 强绑定，难以修改，也无法对接本地模型或其他模型运行。

**Nano Claude Code** 用约 10K 行可读的 Python 代码重写了同样的核心循环：保留你真正需要的部分，去掉你不需要的部分。更详细的分析见这里（Nano Claude code v3.03），[英文版](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_en.md) 和 [中文版](https://github.com/SafeRL-Lab/nano-claude-code/blob/main/docs/comparison_claude_code_vs_nano_v3.03_cn.md)。

### 一目了然

| 维度 | Claude Code（TypeScript） | Nano Claude Code（Python） |
|-----------|--------------------------|---------------------------|
| 语言 | TypeScript + React/Ink | Python 3.8+ |
| 源文件数量 | ~1,332 个 TS/TSX 文件 | 51 个 Python 文件 |
| 代码行数 | ~283K | ~11.6K |
| 内置工具 | 44+ | 25 |
| Slash 命令 | 88 | 20 |
| 语音输入 | Anthropic 专有 WebSocket（需要 OAuth） | 本地 Whisper / OpenAI API —— 可离线，无需订阅 |
| 模型提供方 | 仅 Anthropic | 7+（Anthropic · OpenAI · Gemini · Kimi · Qwen · DeepSeek · Ollama · …） |
| 本地模型 | 不支持 | 支持 —— Ollama、LM Studio、vLLM、任意 OpenAI-compatible endpoint |
| 是否需要构建步骤 | 需要（Bun + esbuild） | 不需要 —— 直接运行 `python nano_claude.py`（或安装后使用 `nano_claude`） |
| 运行时可扩展性 | 封闭（编译时） | 开放 —— 运行时 `register_tool()`，Markdown skills，git plugins |
| 任务依赖图 | 不支持 | 支持 —— `task/` 包中的 `blocks` / `blocked_by` 边 |

### Claude Code 的优势

- **UI 质量** —— 基于 React/Ink 的组件树，支持流式渲染、细粒度 diff 可视化和对话框系统。
- **工具广度** —— 44 个工具，包括 `RemoteTrigger`、`EnterWorktree` 等更偏 UI 集成的工具。
- **企业级特性** —— MDM 管理配置、团队权限同步、OAuth、钥匙串存储、GrowthBook 特性开关。
- **AI 驱动的记忆抽取** —— `extractMemories` 服务会在无显式工具调用的情况下主动从对话中抽取知识。
- **生产可靠性** —— 单文件分发 `cli.js`、全面的测试覆盖、版本锁定发布。

### Nano Claude Code 的优势

- **多 provider** —— 使用 `--model` 或 `/model`，可以在 Claude、GPT-4o、Gemini 2.5 Pro、DeepSeek、Qwen 或本地 Llama 模型之间切换 —— 无需重新编译。
- **本地模型支持** —— 通过 Ollama、LM Studio 或任意 vLLM 托管模型完全离线运行。
- **源码易读** —— 完整的 agent loop 仅 174 行（`agent.py`）。任何 Python 开发者都可以在几分钟内读懂、fork 并扩展它。
- **零构建** —— `pip install -r requirements.txt` 即可运行。修改代码后立即生效。
- **动态可扩展** —— 可在运行时用 `register_tool(ToolDef(...))` 注册新工具，可从 git URL 安装 skill 包，也可接入任意 MCP 服务器。
- **任务依赖图** —— `TaskCreate` / `TaskUpdate` 支持 `blocks` / `blocked_by` 边，用于结构化多步规划（Claude Code 不具备）。
- **双层上下文压缩** —— 基于规则的 snip + AI 摘要，可通过 `preserve_last_n_turns` 配置。
- **Notebook 编辑** —— `NotebookEdit` 直接操作 `.ipynb` JSON（替换/插入/删除单元格），无需内核。
- **无需 LSP 服务器的诊断** —— `GetDiagnostics` 对 Python 使用 pyright → mypy → flake8 → py_compile，对其他语言使用 tsc/shellcheck，零配置。
- **离线语音输入** —— `/voice` 通过 `sounddevice`/`arecord`/SoX 录音，使用本地 `faster-whisper` 转写（无需 API Key、无需订阅），并自动提交。来自 git 分支和项目文件的关键词还能提升代码术语识别准确率。
- **云端会话同步** —— `/cloudsave` 将对话备份到私有 GitHub Gists，无额外依赖；在任意机器上都可用 `/cloudsave load <id>` 恢复。
- **主动式后台监控** —— `/proactive 5m` 会启动一个哨兵守护线程，在用户无操作一段时间后自动唤醒代理，从而支持持续监控循环、定时检查，甚至交易机器人，无需用户再次输入。
- **Rich Live 流式渲染** —— 安装 `rich` 后，回复会以原位更新的 Markdown 方式流式显示（不会输出重复的原始文本），并能干净地与工具调用交错显示。
- **原生 Ollama 推理显示** —— 本地推理模型（deepseek-r1、qwen3、gemma4）会通过 `ThinkingChunk` 事件将 `<think>` token 直接流式输出到终端；通过 `/verbose` 和 `/thinking` 启用。
- **原生 Ollama 视觉** —— `/image [prompt]` 会抓取剪贴板，并通过 Ollama 原生图像 API 发送给本地视觉模型（llava、gemma4、llama3.2-vision）。无需云服务。
- **可靠的多行粘贴** —— Bracketed Paste Mode（`ESC[?2004h`）会将任意粘贴内容 —— 代码块、多段提示、长 diff —— 作为单个 turn 收集，零延迟且不会产生空行伪影。
- **Rich Tab 补全** —— 在 `/` 后按 Tab 会显示所有命令的一行描述和子命令提示；对 `/mcp`、`/plugin`、`/tasks`、`/cloudsave` 等都支持子命令 Tab 补全。

### 核心设计差异

**Agent loop** —— Nano 使用 Python generator，`yield` 出带类型的事件（`TextChunk`、`ToolStart`、`ToolEnd`、`TurnDone`）。完整循环都在一个文件中，便于加入 hook、自定义渲染器或日志。

**工具注册** —— 每个工具都是一个 `ToolDef(name, schema, func, read_only, concurrent_safe)` 数据类。任意模块都可以在 import 时调用 `register_tool()`；MCP 服务器、插件和技能都使用同一套机制。

**上下文压缩**

| | Claude Code | Nano Claude Code |
|-|-------------|-----------------|
| 触发方式 | 精确 token 计数 | `len / 3.5` 估算，在 70% 时触发 |
| 第一层 | — | Snip：截断旧工具输出（无 API 成本） |
| 第二层 | AI summarization | 对旧消息进行 AI 摘要 |
| 控制方式 | 系统管理 | `preserve_last_n_turns` 参数 |

**记忆** —— Claude Code 的 `extractMemories` 服务会让模型主动暴露事实。Nano 的 `memory/` 包是工具驱动的：模型通过显式调用 `MemorySave` 来保存记忆，因此更可预测、更可审计。每条记忆现在都带有 `confidence`、`source`、`last_used_at` 和 `conflict_group` 元数据；检索会按 confidence × recency 重新排序；`/memory consolidate` 提供手动整合流程，不会在后台静默修改记忆。

### 谁适合使用 Nano Claude Code

- 想把**本地模型或非 Anthropic 模型**作为编程助手使用的开发者。
- 研究**agentic coding assistant 工作机制**的研究者 —— 整个系统在一屏内即可看清。
- 需要一个**可 hack 的基线系统**，来加入私有工具、自定义权限策略或专用 agent 类型的团队。
- 想要 Claude Code 风格的生产力，但**不想依赖 Node.js 构建链**的用户。

---

## 特性

| 特性 | 说明 |
|---|---|
| 多 provider | Anthropic · OpenAI · Gemini · Kimi · Qwen · Zhipu · DeepSeek · Ollama · LM Studio · 自定义 endpoint |
| 交互式 REPL | readline 历史记录、带描述和子命令提示的 Tab 补全；Bracketed Paste Mode 支持可靠多行粘贴 |
| Agent loop | Streaming API + 自动工具调用循环 |
| 25 个内置工具 | Read · Write · Edit · Bash · Glob · Grep · WebFetch · WebSearch · **NotebookEdit** · **GetDiagnostics** · MemorySave · MemoryDelete · MemorySearch · MemoryList · Agent · SendMessage · CheckAgentResult · ListAgentTasks · ListAgentTypes · Skill · SkillList · AskUserQuestion · TaskCreate/Update/Get/List · **SleepTimer** · *(MCP + plugin 工具会在启动时自动加入)* |
| MCP 集成 | 连接任意 MCP server（stdio/SSE/HTTP），工具自动注册并可被 Claude 调用 |
| 插件系统 | 可从 git URL 或本地路径安装/卸载/启用/禁用/更新插件；支持多作用域（user/project）；包含推荐引擎 |
| AskUserQuestion | Claude 可在任务中途暂停并向用户发起澄清问题，可附带编号选项 |
| 任务管理 | TaskCreate/Update/Get/List 工具；顺序 ID；依赖边；元数据；持久化到 `.nano_claude/tasks.json`；`/tasks` REPL 命令 |
| Diff 视图 | 对 Edit 和 Write 显示 git 风格红绿 diff |
| 上下文压缩 | 自动压缩长对话以保持在模型上下文限制内 |
| 持久记忆 | 双作用域记忆（user + project），支持 4 类记忆、confidence/source 元数据、冲突检测、带时间衰减的检索、`last_used_at` 跟踪，以及通过 `/memory consolidate` 进行自动抽取 |
| 多代理 | 可生成带类型的子代理（coder/reviewer/researcher/…），支持 git worktree 隔离与后台模式 |
| 技能 | 内置 `/commit` · `/review` + 自定义 markdown skills，支持参数替换及 fork/inline 执行 |
| 插件工具 | 通过 `tool_registry.py` 注册自定义工具 |
| 权限系统 | `auto` / `accept-all` / `manual` 模式 |
| 19 个 slash 命令 | `/model` · `/config` · `/save` · `/cost` · `/memory` · `/skills` · `/agents` · `/voice` · `/proactive` · … |
| 语音输入 | 录音 → 转写 → 自动提交。后端：`sounddevice` / `arecord` / SoX + `faster-whisper` / `openai-whisper` / OpenAI API。完全支持离线。 |
| Brainstorm | `/brainstorm [topic]` 为特定主题生成 N 个适配专家人格（2–100，默认 5，交互选择），执行迭代式辩论，将结果保存至 `brainstorm_outputs/`，并综合生成 Master Plan。 |
| 视觉输入 | `/image [prompt]` 抓取剪贴板图片并发送给本地视觉模型（Ollama `llava`、`gemma4`、`llama3.2-vision`）。需要 `pip install nano-claude-code[vision]`；Linux 还需 `xclip`。 |
| 主动式监控 | `/proactive [duration]` 启动后台哨兵守护线程；代理会在无操作一段时间后自动唤醒，实现无需用户提示的持续监控循环 |
| Rich Live 流式渲染 | 安装 `rich` 后，回复会以原位更新 Markdown 的形式显示。SSH 会话中会自动关闭，以避免重复输出；可通过 `/config rich_live=false` 覆盖。 |
| 上下文注入 | 自动加载 `CLAUDE.md`、git 状态、当前工作目录、持久记忆 |
| 会话持久化 | 退出时自动保存到 `daily/YYYY-MM-DD/`（每日上限）+ `history.json`（主历史）+ `session_latest.json`（用于 `/resume`）；会话文件包含 `session_id` 与 `saved_at` 元数据；`/load` 按日期分组 |
| 云同步 | `/cloudsave` 将会话同步到私有 GitHub Gists；退出时自动同步；可通过 Gist ID 从云端加载。无新增依赖（标准库 `urllib`）。 |
| Extended Thinking | 可为 Claude 模型开启/关闭；本地 Ollama 推理模型（deepseek-r1、qwen3、gemma4）支持原生 `<think>` 流式显示 |
| 成本跟踪 | token 使用量 + 预估美元成本 |
| 非交互模式 | `--print` 标志，适合脚本 / CI |

---

## 支持的模型

### 闭源模型（API）

| Provider | 模型 | 上下文长度 | 优势 | API Key 环境变量 |
|---|---|---|---|---|
| **Anthropic** | `claude-opus-4-6` | 200k | 能力最强，适合复杂推理 | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-sonnet-4-6` | 200k | 速度与质量平衡 | `ANTHROPIC_API_KEY` |
| **Anthropic** | `claude-haiku-4-5-20251001` | 200k | 快速，成本低 | `ANTHROPIC_API_KEY` |
| **OpenAI** | `gpt-4o` | 128k | 多模态和编程能力强 | `OPENAI_API_KEY` |
| **OpenAI** | `gpt-4o-mini` | 128k | 快、便宜 | `OPENAI_API_KEY` |
| **OpenAI** | `o3-mini` | 200k | 强推理 | `OPENAI_API_KEY` |
| **OpenAI** | `o1` | 200k | 高级推理 | `OPENAI_API_KEY` |
| **Google** | `gemini-2.5-pro-preview-03-25` | 1M | 长上下文，多模态 | `GEMINI_API_KEY` |
| **Google** | `gemini-2.0-flash` | 1M | 速度快，长上下文 | `GEMINI_API_KEY` |
| **Google** | `gemini-1.5-pro` | 2M | 最大上下文窗口 | `GEMINI_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-8k` | 8k | 中文和英文表现好 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-32k` | 32k | 中文和英文表现好 | `MOONSHOT_API_KEY` |
| **Moonshot (Kimi)** | `moonshot-v1-128k` | 128k | 长上下文 | `MOONSHOT_API_KEY` |
| **Alibaba (Qwen)** | `qwen-max` | 32k | Qwen 质量最佳 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-plus` | 128k | 平衡 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwen-turbo` | 1M | 快、便宜 | `DASHSCOPE_API_KEY` |
| **Alibaba (Qwen)** | `qwq-32b` | 32k | 推理强 | `DASHSCOPE_API_KEY` |
| **Zhipu (GLM)** | `glm-4-plus` | 128k | GLM 质量最佳 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4` | 128k | 通用用途 | `ZHIPU_API_KEY` |
| **Zhipu (GLM)** | `glm-4-flash` | 128k | 有免费层 | `ZHIPU_API_KEY` |
| **DeepSeek** | `deepseek-chat` | 64k | 编程能力强 | `DEEPSEEK_API_KEY` |
| **DeepSeek** | `deepseek-reasoner` | 64k | Chain-of-thought 推理 | `DEEPSEEK_API_KEY` |

### 开源模型（本地，通过 Ollama）

| 模型 | 参数规模 | 优势 | 拉取命令 |
|---|---|---|---|
| `llama3.3` | 70B | 通用能力强，推理能力好 | `ollama pull llama3.3` |
| `llama3.2` | 3B / 11B | 轻量级 | `ollama pull llama3.2` |
| `qwen2.5-coder` | 7B / 32B | **最适合代码任务** | `ollama pull qwen2.5-coder` |
| `qwen2.5` | 7B / 72B | 中文与英文 | `ollama pull qwen2.5` |
| `deepseek-r1` | 7B–70B | 推理、数学 | `ollama pull deepseek-r1` |
| `deepseek-coder-v2` | 16B | 编程 | `ollama pull deepseek-coder-v2` |
| `mistral` | 7B | 快速、高效 | `ollama pull mistral` |
| `mixtral` | 8x7B | 强大的 MoE 模型 | `ollama pull mixtral` |
| `phi4` | 14B | 微软模型，推理强 | `ollama pull phi4` |
| `gemma3` | 4B / 12B / 27B | Google 开源模型 | `ollama pull gemma3` |
| `codellama` | 7B / 34B | 代码生成 | `ollama pull codellama` |
| `llava` | 7B / 13B | **视觉** —— 图像理解 | `ollama pull llava` |
| `llama3.2-vision` | 11B | **视觉** —— 多模态推理 | `ollama pull llama3.2-vision` |

> **注意：** 工具调用需要模型支持 function calling。推荐的本地模型：`qwen2.5-coder`、`llama3.3`、`mistral`、`phi4`。

> **推理模型：** `deepseek-r1`、`qwen3` 和 `gemma4` 支持原生 `<think>` block 流式显示。使用 `/verbose` 和 `/thinking` 可在终端查看思考过程。注意：当模型接收到较大的系统提示（如 nano-claude 的 25 个工具 schema）时，可能会抑制其 thinking 阶段，以避免破坏预期的 JSON 格式 —— 这是模型行为，不是 bug。

---

## 安装

### 推荐：使用 `uv` 安装为全局命令

[uv](https://docs.astral.sh/uv/) 会将 `nano_claude` 安装到隔离环境中，并加入你的 PATH，这样你就可以在任意位置运行：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/SafeRL-Lab/nano-claude-code
cd nano-claude-code
uv tool install .
```

安装后，`nano_claude` 将成为全局命令：

```bash
nano_claude                        # 启动 REPL
nano_claude --model gpt-4o         # 选择模型
nano_claude -p "explain this"      # 非交互模式
```

拉取新代码后更新：

```bash
uv tool install . --reinstall
```

卸载：

```bash
uv tool uninstall nano-claude-code
```

### 另一种方式：直接从仓库运行

```bash
git clone https://github.com/SafeRL-Lab/nano-claude-code
cd nano-claude-code

pip install -r requirements.txt
# 或手动安装（sounddevice 可选，仅 /voice 需要）
pip install anthropic openai httpx rich
pip install sounddevice  # 可选：语音输入

python nano_claude.py
```

---

## 使用：闭源 API 模型

### Anthropic Claude

在 [console.anthropic.com](https://console.anthropic.com) 获取 API Key。

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...

# 默认模型（claude-opus-4-6）
nano_claude

# 选择特定模型
nano_claude --model claude-sonnet-4-6
nano_claude --model claude-haiku-4-5-20251001

# 开启 Extended Thinking
nano_claude --model claude-opus-4-6 --thinking --verbose
```

### OpenAI GPT

在 [platform.openai.com](https://platform.openai.com) 获取 API Key。

```bash
export OPENAI_API_KEY=sk-...

nano_claude --model gpt-4o
nano_claude --model gpt-4o-mini
nano_claude --model gpt-4.1-mini
nano_claude --model o3-mini
```

### Google Gemini

在 [aistudio.google.com](https://aistudio.google.com) 获取 API Key。

```bash
export GEMINI_API_KEY=AIza...

nano_claude --model gemini/gemini-2.0-flash
nano_claude --model gemini/gemini-1.5-pro
nano_claude --model gemini/gemini-2.5-pro-preview-03-25
```

### Kimi（Moonshot AI）

在 [platform.moonshot.cn](https://platform.moonshot.cn) 获取 API Key。

```bash
export MOONSHOT_API_KEY=sk-...

nano_claude --model kimi/moonshot-v1-32k
nano_claude --model kimi/moonshot-v1-128k
```

### Qwen（Alibaba DashScope）

在 [dashscope.aliyun.com](https://dashscope.aliyun.com) 获取 API Key。

```bash
export DASHSCOPE_API_KEY=sk-...

nano_claude --model qwen/Qwen3.5-Plus
nano_claude --model qwen/Qwen3-MAX
nano_claude --model qwen/Qwen3.5-Flash
```

### Zhipu GLM

在 [open.bigmodel.cn](https://open.bigmodel.cn) 获取 API Key。

```bash
export ZHIPU_API_KEY=...

nano_claude --model zhipu/glm-4-plus
nano_claude --model zhipu/glm-4-flash   # 免费层
```

### DeepSeek

在 [platform.deepseek.com](https://platform.deepseek.com) 获取 API Key。

```bash
export DEEPSEEK_API_KEY=sk-...

nano_claude --model deepseek/deepseek-chat
nano_claude --model deepseek/deepseek-reasoner
```

---

## 使用：开源模型（本地）

### 方案 A — Ollama（推荐）

Ollama 可以零配置在本地运行模型。无需 API Key。

**步骤 1：安装 Ollama**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或从 https://ollama.com/download 下载
```

**步骤 2：拉取模型**

```bash
# 最适合编码（推荐）
ollama pull qwen2.5-coder          # 4.7 GB（7B）
ollama pull qwen2.5-coder:32b      # 19 GB（32B）

# 通用用途
ollama pull llama3.3               # 42 GB（70B）
ollama pull llama3.2               # 2.0 GB（3B）

# 推理
ollama pull deepseek-r1            # 4.7 GB（7B）
ollama pull deepseek-r1:32b        # 19 GB（32B）

# 其他
ollama pull phi4                   # 9.1 GB（14B）
ollama pull mistral                # 4.1 GB（7B）
```

**步骤 3：启动 Ollama 服务**（macOS 上会自动运行；Linux 上需手动启动）

```bash
ollama serve     # 启动在 http://localhost:11434
```

**步骤 4：运行 nano claude**

```bash
nano_claude --model ollama/qwen2.5-coder
nano_claude --model ollama/llama3.3
nano_claude --model ollama/deepseek-r1
```

或者

```bash
python nano_claude.py --model ollama/qwen2.5-coder
python nano_claude.py --model ollama/llama3.3
python nano_claude.py --model ollama/deepseek-r1
python nano_claude.py --model ollama/qwen3.5:35b
```

**查看本地可用模型：**

```bash
ollama list
```

然后使用列表中的任意模型：

```bash
nano_claude --model ollama/<model-name>
```

---

### 方案 B — LM Studio

LM Studio 提供图形界面，用于下载和运行模型，并内置 OpenAI-compatible 服务。

**步骤 1：** 下载并安装 [LM Studio](https://lmstudio.ai)。

**步骤 2：** 在 LM Studio 中搜索并下载模型（GGUF 格式）。

**步骤 3：** 进入 **Local Server** 标签页 → 点击 **Start Server**（默认端口：1234）。

**步骤 4：**

```bash
nano_claude --model lmstudio/<model-name>
# 例如：
nano_claude --model lmstudio/phi-4-GGUF
nano_claude --model lmstudio/qwen2.5-coder-7b
```

模型名称应与 LM Studio 服务状态栏中显示的名称一致。

---

### 方案 C — vLLM / 自托管 OpenAI-Compatible Server

适用于暴露 OpenAI-compatible API 的自托管推理服务（如 vLLM、TGI、llama.cpp server 等）：

方案 C 快速开始：  
步骤 1：启动 vllm：
 ```
CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 0.0.0.0 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser hermes
```


 步骤 2：启动 nano claude：
```
  export CUSTOM_BASE_URL=http://localhost:8000/v1
  export CUSTOM_API_KEY=none
  nano_claude --model custom/Qwen/Qwen2.5-Coder-7B-Instruct
```


```bash
# 示例：vLLM 部署 Qwen2.5-Coder-32B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8000

# 然后让 nano claude 指向你的服务：
nano_claude
```

在 REPL 中：

```
/config custom_base_url=http://localhost:8000/v1
/config custom_api_key=token-abc123    # 如无鉴权可跳过
/model custom/Qwen2.5-Coder-32B-Instruct
```

或者通过环境变量：

```bash
export CUSTOM_BASE_URL=http://localhost:8000/v1
export CUSTOM_API_KEY=token-abc123

nano_claude --model custom/Qwen2.5-Coder-32B-Instruct
```

如果是远程 GPU 服务器：

```bash
/config custom_base_url=http://192.168.1.100:8000/v1
/model custom/your-model-name
```

---

## 模型名称格式

支持三种等价格式：

```bash
# 1. 按前缀自动检测（适用于常见模型）
nano_claude --model gpt-4o
nano_claude --model gemini-2.0-flash
nano_claude --model deepseek-chat

# 2. 显式 provider 前缀 + 斜杠
nano_claude --model ollama/qwen2.5-coder
nano_claude --model kimi/moonshot-v1-128k

# 3. 显式 provider 前缀 + 冒号（也可用）
nano_claude --model kimi:moonshot-v1-32k
nano_claude --model qwen:qwen-max
```

**自动检测规则：**

| 模型前缀 | 自动识别的 provider |
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

## CLI 参考

```
nano_claude [OPTIONS] [PROMPT]
# 或：python nano_claude.py [OPTIONS] [PROMPT]

选项：
  -p, --print          非交互模式：执行 prompt 后退出
  -m, --model MODEL    覆盖模型（例如 gpt-4o、ollama/llama3.3）
  --accept-all         自动批准所有操作（不弹权限提示）
  --verbose            显示 thinking blocks 和每轮 token 数
  --thinking           启用 Extended Thinking（仅 Claude）
  --version            打印版本并退出
  -h, --help           显示帮助
```

**示例：**

```bash
# 使用默认模型进入交互式 REPL
nano_claude

# 启动时切换模型
nano_claude --model gpt-4o
nano_claude -m ollama/deepseek-r1:32b

# 非交互 / 脚本模式
nano_claude --print "Write a Python fibonacci function"
nano_claude -p "Explain the Rust borrow checker in 3 sentences" -m gemini/gemini-2.0-flash

# CI / 自动化（不弹权限提示）
nano_claude --accept-all --print "Initialize a Python project with pyproject.toml"

# 调试模式（查看 token + thinking）
nano_claude --thinking --verbose
```

---

## Slash 命令（REPL）

输入 `/` 并按 **Tab** 可以查看所有命令及其描述。继续输入可过滤，再按 Tab 自动补全。输入命令名后再次按 **Tab**，可查看其子命令（例如 `/plugin ` → `install`、`uninstall`、`enable` 等）。

| 命令 | 说明 |
|---|---|
| `/help` | 显示全部命令 |
| `/clear` | 清空对话历史 |
| `/model` | 显示当前模型 + 列出所有可用模型 |
| `/model <name>` | 切换模型（立即生效） |
| `/config` | 显示当前全部配置项 |
| `/config key=value` | 设置配置项（持久化到磁盘） |
| `/save` | 保存会话（按时间戳自动命名） |
| `/save <filename>` | 将会话保存为指定文件名 |
| `/load` | 交互式列表，按日期分组；可输入编号、`1,2,3` 合并，或 `H` 加载完整历史 |
| `/load <filename>` | 按文件名加载已保存会话 |
| `/resume` | 恢复最近一次自动保存的会话（`mr_sessions/session_latest.json`） |
| `/resume <filename>` | 从 `mr_sessions/`（或绝对路径）加载指定文件 |
| `/history` | 打印完整对话历史 |
| `/context` | 显示消息数量和 token 估计 |
| `/cost` | 显示 token 用量和预估美元成本 |
| `/verbose` | 切换 verbose 模式（token + thinking） |
| `/thinking` | 切换 Extended Thinking（仅 Claude） |
| `/permissions` | 显示当前权限模式 |
| `/permissions <mode>` | 设置权限模式：`auto` / `accept-all` / `manual` |
| `/cwd` | 显示当前工作目录 |
| `/cwd <path>` | 切换当前工作目录 |
| `/memory` | 列出全部持久记忆 |
| `/memory <query>` | 按关键词搜索记忆（按 confidence × recency 排序） |
| `/memory consolidate` | 从当前会话中用 AI 抽取最多 3 条长期洞见 |
| `/skills` | 列出可用技能 |
| `/agents` | 显示子代理任务状态 |
| `/mcp` | 列出已配置的 MCP 服务器及其工具 |
| `/mcp reload` | 重连所有 MCP 服务器并刷新工具 |
| `/mcp reload <name>` | 重连单个 MCP 服务器 |
| `/mcp add <name> <cmd> [args]` | 向用户配置中添加一个 stdio MCP server |
| `/mcp remove <name>` | 从用户配置中移除一个 server |
| `/voice` | 录音，用 Whisper 转写后自动提交为 prompt |
| `/image [prompt]` | 捕获剪贴板图像并发送给视觉模型，可附带提示词 |
| `/voice status` | 显示录音与 STT 后端是否可用 |
| `/voice lang <code>` | 设置 STT 语言（例如 `zh`、`en`、`ja`；`auto` 为自动检测） |
| `/proactive` | 显示当前主动轮询状态（开/关及间隔） |
| `/proactive <duration>` | 开启后台哨兵轮询（例如 `5m`、`30s`、`1h`） |
| `/proactive off` | 关闭后台轮询 |
| `/cloudsave setup <token>` | 配置用于 Gist 同步的 GitHub Personal Access Token |
| `/cloudsave` | 将当前会话上传到私有 GitHub Gist |
| `/cloudsave push [desc]` | 上传时附加可选描述 |
| `/cloudsave auto on\|off` | 切换 `/exit` 时自动上传 |
| `/cloudsave list` | 列出你的 nano-claude-code Gists |
| `/cloudsave load <gist_id>` | 从 Gist 下载并恢复会话 |
| `/brainstorm` | 运行多人格 AI brainstorming；会提示选择 agent 数量（2–100，默认 5） |
| `/brainstorm <topic>` | 针对指定主题进行 brainstorming；会提示选择 agent 数量 |
| `/exit` / `/quit` | 退出 |

**在会话中切换模型：**

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

## 配置 API Keys

### 方法 1：环境变量（推荐）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=AIza...
export MOONSHOT_API_KEY=sk-...       # Kimi
export DASHSCOPE_API_KEY=sk-...      # Qwen
export ZHIPU_API_KEY=...             # Zhipu GLM
export DEEPSEEK_API_KEY=sk-...       # DeepSeek
```

### 方法 2：在 REPL 中设置（会持久保存）

```
/config anthropic_api_key=sk-ant-...
/config openai_api_key=sk-...
/config gemini_api_key=AIza...
/config kimi_api_key=sk-...
/config qwen_api_key=sk-...
/config zhipu_api_key=...
/config deepseek_api_key=sk-...
```

这些 Key 会保存到 `~/.nano_claude/config.json`，下次启动时会自动加载。

### 方法 3：直接编辑配置文件

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

## 权限系统

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 只读操作始终允许。执行 Bash 命令和写文件前会提示。 |
| `accept-all` | 从不提示。所有操作自动执行。 |
| `manual` | 每一个操作都要提示，包括读取。 |

**提示示例：**

```
  Allow: Run: git commit -am "fix bug"  [y/N/a(ccept-all)]
```

- `y` —— 仅批准这一次操作
- `n` 或直接回车 —— 拒绝
- `a` —— 批准并将本次会话切换到 `accept-all`

**在 `auto` 模式下始终自动批准的命令：**  
`ls`、`cat`、`head`、`tail`、`wc`、`pwd`、`echo`、`git status`、`git log`、`git diff`、`git show`、`find`、`grep`、`rg`、`python`、`node`、`pip show`、`npm list` 以及其他只读 shell 命令。

---

## 内置工具

### 核心工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Read` | 按行号读取文件 | `file_path`, `limit`, `offset` |
| `Write` | 创建或覆盖文件（显示 diff） | `file_path`, `content` |
| `Edit` | 精确字符串替换（显示 diff） | `file_path`, `old_string`, `new_string`, `replace_all` |
| `Bash` | 执行 shell 命令 | `command`, `timeout`（默认 30s） |
| `Glob` | 通过 glob 模式查找文件 | `pattern`（例如 `**/*.py`）, `path` |
| `Grep` | 在文件中进行正则搜索（如可用则使用 ripgrep） | `pattern`, `path`, `glob`, `output_mode` |
| `WebFetch` | 抓取 URL 并提取文本 | `url`, `prompt` |
| `WebSearch` | 通过 DuckDuckGo 搜索网络 | `query` |

### Notebook 与诊断工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `NotebookEdit` | 编辑 Jupyter notebook（`.ipynb`）单元格 | `notebook_path`, `new_source`, `cell_id`, `cell_type`, `edit_mode`（`replace`/`insert`/`delete`） |
| `GetDiagnostics` | 获取源文件的类 LSP 诊断（Python：pyright/mypy/flake8；JS/TS：tsc/eslint；Shell：shellcheck） | `file_path`, `language`（可选覆盖） |

### 记忆工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `MemorySave` | 保存或更新一条持久记忆 | `name`, `type`, `description`, `content`, `scope` |
| `MemoryDelete` | 按名称删除记忆 | `name`, `scope` |
| `MemorySearch` | 按关键词（或 AI 排序）搜索记忆 | `query`, `scope`, `use_ai`, `max_results` |
| `MemoryList` | 列出所有记忆及其年龄与元数据 | `scope` |

### 子代理工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Agent` | 为任务生成一个子代理 | `prompt`, `subagent_type`, `isolation`, `name`, `model`, `wait` |
| `SendMessage` | 向命名的后台代理发送消息 | `name`, `message` |
| `CheckAgentResult` | 检查后台代理的状态/结果 | `task_id` |
| `ListAgentTasks` | 列出所有活动中和已完成的代理任务 | — |
| `ListAgentTypes` | 列出所有可用代理类型定义 | — |

### 后台与自治工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `SleepTimer` | 安排一个静默后台定时器；触发时会注入自动唤醒提示，使代理可继续监控或执行延后任务 | `seconds` |

### 技能工具

| 工具 | 说明 | 关键参数 |
|---|---|---|
| `Skill` | 在对话中按名称调用某个 skill | `name`, `args` |
| `SkillList` | 列出所有可用 skills 及其触发器和元数据 | — |

### MCP 工具

MCP 工具会从已配置的服务器中自动发现，并以 `mcp__<server>__<tool>` 的形式注册。Claude 可以像使用内置工具一样使用它们。

| 示例工具名 | 来源 |
|---|---|
| `mcp__git__git_status` | `git` server，`git_status` 工具 |
| `mcp__filesystem__read_file` | `filesystem` server，`read_file` 工具 |
| `mcp__myserver__my_action` | 你配置的自定义 server |

> **添加自定义工具：** 参见 [Architecture Guide](docs/architecture.md#tool-registry) 了解如何注册你自己的工具。

---

## 记忆系统

模型可以通过内置记忆系统跨会话记住信息。

### 存储方式

记忆会以独立 markdown 文件形式存储在两个作用域中：

| 作用域 | 路径 | 可见性 |
|---|---|---|
| **User**（默认） | `~/.nano_claude/memory/` | 在所有项目间共享 |
| **Project** | 当前目录下 `.nano_claude/memory/` | 仅当前仓库可见 |

系统会在每次保存或删除记忆时自动重建一个 `MEMORY.md` 索引（≤ 200 行 / 25 KB），并将其注入系统提示中，让模型始终拥有记忆总览。

### 记忆类型

| 类型 | 用途 |
|---|---|
| `user` | 你的角色、偏好、背景 |
| `feedback` | 你希望模型如何表现（纠正和确认） |
| `project` | 正在进行的工作、截止日期、未写入 git 历史的决策 |
| `reference` | 指向外部系统的链接（Linear、Grafana、Slack 等） |

### 记忆文件格式

每条记忆都是一个带 YAML frontmatter 的 markdown 文件：

```markdown
---
name: coding_style
description: Python formatting preferences
type: feedback
created: 2026-04-02
confidence: 0.95
source: user
last_used_at: 2026-04-05
conflict_group: coding_style
---
Prefer 4-space indentation and full type hints in all Python code.
**Why:** user explicitly stated this preference.
**How to apply:** apply to every Python file written or edited.
```

**元数据字段**（新增 —— 自动管理）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `confidence` | `1.0` | 可靠性分数，范围 0–1。用户明确陈述 = 1.0；推断偏好 ≈ 0.8；自动整合 ≈ 0.8 |
| `source` | `user` | 来源：`user` / `model` / `tool` / `consolidator` |
| `last_used_at` | — | 每次该记忆被 MemorySearch 返回时会自动更新 |
| `conflict_group` | — | 将相关记忆分组（例如 `writing_style`）以便冲突跟踪 |

### 冲突检测

当使用 `MemorySave` 保存一个已存在名称、但内容不同的记忆时，系统会在覆盖前报告冲突：

```
Memory saved: 'writing_style' [feedback/user]
⚠ Replaced conflicting memory (was user-sourced, 100% confidence, written 2026-04-01).
  Old content: Prefer formal, academic style...
```

### 排序检索

`MemorySearch` 会按 **confidence × recency**（30 天指数衰减）进行排序，而不是简单按关键词顺序。长时间未使用的记忆会逐渐降低优先级。每次命中也会更新 `last_used_at`，使经常访问的记忆保持更靠前。

```
You: /memory python
  [feedback/user] coding_style [conf:95% src:user]
    Python formatting preferences
    Prefer 4-space indentation and full type hints...
```

### `/memory consolidate` —— 自动抽取长期洞见

完成一段有意义的会话后，运行：

```
[myproject] ❯ /memory consolidate
  Analyzing session for long-term memories…
  ✓ Consolidated 2 memory/memories: user_prefers_direct_answers, avoid_trailing_summaries
```

该命令会把压缩后的会话记录发送给模型，让它识别最多 **3 条** 值得长期保留的洞见（用户偏好、反馈修正、项目决策）。提取出的记忆会以 `confidence: 0.80` 和 `source: consolidator` 保存 —— **永远不会覆盖** 已有且置信度更高的记忆。

适合运行 `/memory consolidate` 的时机：
- 当你连续多次纠正模型行为之后
- 当你在会话中分享了项目背景或关键决策之后
- 当你完成一个任务并形成了明确的规划选择之后

### 示例交互

```
You: Remember that I prefer 4-space indentation and type hints.
AI: [calls MemorySave] Memory saved: 'coding_style' [feedback/user]

You: /memory
  1 memory/memories:
  [feedback  |user   ] coding_style.md
    Python formatting preferences

You: /memory python
  Found 1 relevant memory for 'python':
  [feedback/user] coding_style
    Prefer 4-space indentation and full type hints in all Python code.

You: /memory consolidate
  ✓ Consolidated 1 memory: user_prefers_verbose_commit_messages
```

**过期警告：** 超过 1 天的记忆会显示 `⚠ stale` 提示 —— 关于文件:行号引用或代码状态的说法可能已经过时，使用前请验证。

**AI 排序搜索：** `MemorySearch(query="...", use_ai=true)` 会先让模型按相关性排序候选，再应用 confidence × recency 的再排序。

---

## 技能

技能是可复用的提示模板，用来给模型提供专门能力。系统默认内置两个技能 —— 无需任何设置即可使用。

**内置技能：**

| 触发器 | 说明 |
|---|---|
| `/commit` | 审查已暂存的变更并生成结构良好的 git commit |
| `/review [PR]` | 对代码或 PR diff 进行结构化审查 |

**快速开始 —— 自定义 skill：**

```bash
mkdir -p ~/.nano_claude/skills
```

创建 `~/.nano_claude/skills/deploy.md`：

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

现在就可以这样使用：

```
You: /deploy staging 2.1.0
AI: [deploys version 2.1.0 to staging]
```

**参数替换：**
- `$ARGUMENTS` —— 原始完整参数字符串
- `$ARG_NAME` —— 按命名参数进行位置替换（第一个词 → 第一个参数名）
- 缺失参数会被替换为空字符串

**执行模式：**
- `context: inline`（默认）—— 在当前对话历史中运行
- `context: fork` —— 在隔离的子代理中运行，拥有新的历史；支持 `model` 覆盖

**优先级**（高者优先）：project-level > user-level > built-in

**查看 skills：** `/skills` —— 显示触发器、参数提示、来源和 `when_to_use`

**Skill 搜索路径：**

```
./.nano_claude/skills/     # project-level（覆盖 user-level）
~/.nano_claude/skills/     # user-level
```

---

## 子代理

模型可以生成独立的子代理，并行处理任务。

**内置的专用代理类型：**

| 类型 | 优化方向 |
|---|---|
| `general-purpose` | 研究、探索、多步任务 |
| `coder` | 编写、阅读和修改代码 |
| `reviewer` | 安全性、正确性和代码质量分析 |
| `researcher` | Web 搜索和文档查找 |
| `tester` | 编写和运行测试 |

**基本用法：**
```
You: Search this codebase for all TODO comments and summarize them.
AI: [calls Agent(prompt="...", subagent_type="researcher")]
    Sub-agent reads files, greps for TODOs...
    Result: Found 12 TODOs across 5 files...
```

**后台模式** —— 先启动、不等待，稍后再取回结果：
```
AI: [calls Agent(prompt="run all tests", name="test-runner", wait=false)]
AI: [continues other work...]
AI: [calls CheckAgentResult / SendMessage to follow up]
```

**Git worktree 隔离** —— 子代理可在隔离分支中工作，互不冲突：
```
Agent(prompt="refactor auth module", isolation="worktree")
```
若 worktree 未产生修改，会自动清理；否则会报告对应分支名。

**自定义代理类型** —— 创建 `~/.nano_claude/agents/myagent.md`：
```markdown
---
name: myagent
description: Specialized for X
model: claude-haiku-4-5-20251001
tools: [Read, Grep, Bash]
---
Extra system prompt for this agent type.
```

**查看运行中的代理：** `/agents`

子代理拥有独立的对话历史、共享文件系统，并且最多支持 3 层嵌套。

---

## MCP（Model Context Protocol）

MCP 让你可以连接任意外部工具服务器 —— 本地子进程或远程 HTTP 服务 —— Claude 会自动使用其工具。这与 Claude Code 扩展能力时使用的是同一协议。

### 支持的传输方式

| 传输方式 | 配置 `type` | 说明 |
|---|---|---|
| **stdio** | `"stdio"` | 启动本地子进程（最常见） |
| **SSE** | `"sse"` | HTTP Server-Sent Events 流 |
| **HTTP** | `"http"` | 可流式 POST 的 HTTP（较新服务器） |

### 配置方式

在项目目录中放置 `.mcp.json` 文件，**或者** 编辑 `~/.nano_claude/mcp.json` 来配置用户级服务器。

```json
{
  "mcpServers": {
    "git": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-git"]
    },
    "filesystem": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/tmp"]
    },
    "my-remote": {
      "type": "sse",
      "url": "http://localhost:8080/sse",
      "headers": {"Authorization": "Bearer my-token"}
    }
  }
}
```

配置优先级：`.mcp.json`（项目级）会按 server name 覆盖 `~/.nano_claude/mcp.json`（用户级）。

### 快速开始

```bash
# 安装一个常用 MCP server
pip install uv        # uv 内含 uvx
uvx mcp-server-git --help   # 验证是否可用

# 在 REPL 中添加到用户配置
/mcp add git uvx mcp-server-git

# 或者在项目目录创建 .mcp.json，然后：
/mcp reload
```

### REPL 命令

```
/mcp                          # 列出 servers、它们的工具以及连接状态
/mcp reload                   # 重连所有 server，并刷新工具列表
/mcp reload git               # 重连单个 server
/mcp add myserver uvx mcp-server-x   # 添加 stdio server
/mcp remove myserver          # 从用户配置中移除
```

### Claude 如何使用 MCP 工具

一旦连接成功，Claude 就可以直接调用 MCP 工具：

```
You: What files changed in the last git commit?
AI: [calls mcp__git__git_diff_staged()]
    → shows diff output from the git MCP server
```

工具名格式为 `mcp__<server_name>__<tool_name>`。所有非字母数字和非 `_` 的字符都会自动替换为 `_`。

### 常见 MCP servers

| Server | 安装方式 | 提供能力 |
|---|---|---|
| `mcp-server-git` | `uvx mcp-server-git` | git 操作（status、diff、log、commit） |
| `mcp-server-filesystem` | `uvx mcp-server-filesystem <path>` | 文件读写/列出 |
| `mcp-server-fetch` | `uvx mcp-server-fetch` | HTTP 抓取工具 |
| `mcp-server-postgres` | `uvx mcp-server-postgres <conn-str>` | PostgreSQL 查询 |
| `mcp-server-sqlite` | `uvx mcp-server-sqlite --db-path x.db` | SQLite 查询 |
| `mcp-server-brave-search` | `uvx mcp-server-brave-search` | Brave 网页搜索 |

> 完整服务器列表可在 [modelcontextprotocol.io/servers](https://modelcontextprotocol.io/servers) 浏览。

---

## 插件系统

`plugin/` 包允许你通过 git 仓库或本地目录，为 nano-claude-code 增加额外工具、skills 和 MCP servers。

### 安装插件

```bash
/plugin install my-plugin@https://github.com/user/my-plugin
/plugin install local-plugin@/path/to/local/plugin
```

### 管理插件

```bash
/plugin                   # 列出已安装插件
/plugin enable my-plugin  # 启用一个已禁用的插件
/plugin disable my-plugin # 禁用但不卸载
/plugin disable-all       # 禁用所有插件
/plugin update my-plugin  # 从 git 拉取最新版本
/plugin uninstall my-plugin
/plugin info my-plugin    # 显示清单详情
```

### 插件推荐引擎

```bash
/plugin recommend                    # 根据项目文件自动检测
/plugin recommend "docker database"  # 根据关键词上下文推荐
```

推荐引擎会根据你的上下文，与一个精选插件市场（git-tools、python-linter、docker-tools、sql-tools、test-runner、diagram-tools、aws-tools、web-scraper）进行标签和关键词匹配打分。

### 插件清单（plugin.json）

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Does something useful",
  "author": "you",
  "tags": ["git", "python"],
  "tools": ["tools"],        // 导出 TOOL_DEFS 的 Python 模块
  "skills": ["skills/my.md"],
  "mcp_servers": {},
  "dependencies": ["httpx"]  // pip 包
}
```

也可以用带 YAML frontmatter 的 `PLUGIN.md`。

### 作用域

| Scope | 位置 | 配置文件 |
|-------|----------|--------|
| user（默认） | `~/.nano_claude/plugins/` | `~/.nano_claude/plugins.json` |
| project | `.nano_claude/plugins/` | `.nano_claude/plugins.json` |

使用 `--project` 标志：`/plugin install name@url --project`

---

## AskUserQuestion 工具

Claude 可以在任务中途暂停，并以交互方式向你提问，然后再继续。

**Claude 调用示例：**
```json
{
  "tool": "AskUserQuestion",
  "question": "Which database should I use?",
  "options": [
    {"label": "SQLite", "description": "Simple, file-based"},
    {"label": "PostgreSQL", "description": "Full-featured, requires server"}
  ],
  "allow_freetext": true
}
```

**你在终端中会看到：**
```
❓ Question from assistant:
   Which database should I use?

  [1] SQLite — Simple, file-based
  [2] PostgreSQL — Full-featured, requires server
  [0] Type a custom answer

Your choice (number or text):
```

- 可以按编号选择，也可以直接输入自由文本
- Claude 接收到你的答案后会继续任务
- 5 分钟超时（若未回答，会返回 `"(no answer — timeout)"`）

---

## 任务管理

`task/` 包为 Claude（以及你）提供了结构化任务列表，用于在一个会话中跟踪多步工作。

### Claude 可用的工具

| 工具 | 参数 | 功能 |
|------|-----------|--------------|
| `TaskCreate` | `subject`, `description`, `active_form?`, `metadata?` | 创建任务；返回 `#id created: subject` |
| `TaskUpdate` | `task_id`, `subject?`, `description?`, `status?`, `owner?`, `add_blocks?`, `add_blocked_by?`, `metadata?` | 更新任意字段；`status='deleted'` 时移除任务 |
| `TaskGet` | `task_id` | 返回单个任务的完整详情 |
| `TaskList` | _(none)_ | 列出所有任务及其状态图标与待解决 blocker |

**有效状态：** `pending` → `in_progress` → `completed` / `cancelled` / `deleted`

### 依赖边

```
TaskUpdate(task_id="3", add_blocked_by=["1","2"])
# 任务 3 现在被任务 1 和 2 阻塞。
# 反向边会自动设置：任务 1 和 2 的 "blocks" 列表中会加入任务 3。
```

已完成任务会被视为已解除阻塞 —— `TaskList` 不再把它们当作依赖障碍。

### 持久化

任务在每次变更后都会保存到当前工作目录下的 `.nano_claude/tasks.json`，并在首次访问时自动加载。

### REPL 命令

```
/tasks                    列出所有任务
/tasks create <subject>   快速创建任务
/tasks start <id>         标记为 in_progress
/tasks done <id>          标记为 completed
/tasks cancel <id>        标记为 cancelled
/tasks delete <id>        删除任务
/tasks get <id>           显示完整详情
/tasks clear              删除所有任务
```

### Claude 的典型工作流

```
User: implement the login feature

Claude:
  TaskCreate(subject="Design auth schema", description="JWT vs session")  → #1
  TaskCreate(subject="Write login endpoint", description="POST /auth/login") → #2
  TaskCreate(subject="Write tests", description="Unit + integration") → #3
  TaskUpdate(task_id="2", add_blocked_by=["1"])
  TaskUpdate(task_id="3", add_blocked_by=["2"])

  TaskUpdate(task_id="1", status="in_progress", active_form="Designing schema")
  ... (开始执行) ...
  TaskUpdate(task_id="1", status="completed")
  TaskList()  → task 2 is now unblocked
  ...
```

---

## 语音输入

Nano Claude Code v3.05 增加了一个完全离线的语音转提示词流程。你只需要说出请求，它会被转写并像你手动输入一样提交。

### 快速开始

```bash
# 1. 安装一个录音后端（任选其一）
pip install sounddevice        # 推荐：跨平台、无需额外二进制
# sudo apt install alsa-utils  # Linux 下 arecord 备用
# sudo apt install sox         # SoX rec 备用

# 2. 安装本地 STT 后端（推荐 —— 离线、无需 API key）
pip install faster-whisper numpy

# 3. 启动 Nano Claude Code 并开始说话
nano_claude
[myproject] ❯ /voice
  🎙  Listening… (speak now, auto-stops on silence, Ctrl+C to cancel)
  🎙  ████
✓  Transcribed: "fix the authentication bug in user.py"
[auto-submitting…]
```

### STT 后端（按顺序尝试）

| 后端 | 安装方式 | 说明 |
|---|---|---|
| `faster-whisper` | `pip install faster-whisper` | **推荐** —— 本地、离线、最快，可选 GPU |
| `openai-whisper` | `pip install openai-whisper` | 本地、离线、OpenAI 原始模型 |
| OpenAI Whisper API | 设置 `OPENAI_API_KEY` | 云端，需要网络和 API Key |

可使用 `NANO_CLAUDE_WHISPER_MODEL` 覆盖 Whisper 模型大小（默认：`base`）：

```bash
export NANO_CLAUDE_WHISPER_MODEL=small   # 更准，但更慢
export NANO_CLAUDE_WHISPER_MODEL=tiny    # 最快，最轻量
```

### 录音后端（按顺序尝试）

| 后端 | 安装方式 | 说明 |
|---|---|---|
| `sounddevice` | `pip install sounddevice` | **推荐** —— 跨平台、原生 Python |
| `arecord` | `sudo apt install alsa-utils` | Linux ALSA，无需 pip |
| `sox rec` | `sudo apt install sox` / `brew install sox` | 内置静音检测 |

### 关键词增强

每次录音前，Nano 会从以下来源提取编程相关词汇：
- **Git 分支名**（例如 `feat/voice-input` → “feat”、“voice”、“input”）
- **项目根目录名称**（例如 “nano-claude-code”）
- **近期源文件文件名**（例如 `authentication_handler.py` → “authentication”、“handler”）
- **全局编程术语**：`MCP`、`grep`、`TypeScript`、`OAuth`、`regex`、`gRPC` 等

这些词会作为 Whisper 的 `initial_prompt` 传入，使 STT 引擎更倾向于输出正确的代码术语拼写。

### 命令

| 命令 | 说明 |
|---|---|
| `/voice` | 录音并自动把转写结果作为下一条 prompt 提交 |
| `/voice status` | 显示可用的录音和 STT 后端 |
| `/voice lang <code>` | 设置转写语言（`en`、`zh`、`ja`、`de`、`fr` 等，默认：`auto`） |

### 与 Claude Code 的对比

| | Claude Code | Nano Claude Code v3.05 |
|---|---|---|
| STT 服务 | Anthropic 私有 WebSocket（`voice_stream`） | `faster-whisper` / `openai-whisper` / OpenAI API |
| 是否需要 Anthropic OAuth | 是 | **否** |
| 是否可离线运行 | 否 | **是**（使用本地 Whisper） |
| 关键词提示 | Deepgram `keyterms` 参数 | Whisper `initial_prompt`（git + 文件 + 词汇） |
| 语言支持 | 服务端白名单语言代码 | Whisper 支持的任意语言 |

---

## Brainstorm

`/brainstorm` 会围绕你的项目展开一个结构化的多人格 AI 辩论，并最终将所有观点综合为一份可执行计划。

### 工作流程

1. **上下文快照** —— 读取当前工作目录中的 `README.md`、`CLAUDE.md` 和根目录文件列表。
2. **代理数量** —— 系统会提示你选择代理数量（2–100，默认 5）。直接回车即可使用默认值。
3. **动态生成人格** —— 模型会根据你的主题生成 N 个合适的专家角色。软件类主题会产生架构师、工程师；地缘政治会产生分析师、外交官、经济学家；商业会产生策略师、市场专家等。若生成失败，会回退到内置技术人格。
4. **代理依次辩论**，每个代理都基于前一个代理的输出继续展开。
5. **输出保存** 到当前目录下的 `brainstorm_outputs/brainstorm_YYYYMMDD_HHMMSS.md`。
6. **综合** —— 主代理读取保存文件，并生成按优先级排序的 Master Plan。

**不同主题示例人格：**

| 主题 | 生成的人格示例 |
|---|---|
| 软件架构 | 🏗️ 架构师 · 💡 产品创新者 · 🛡️ 安全工程师 · 🔧 代码质量负责人 · ⚡ 性能专家 |
| 美伊地缘政治 | 🌍 地缘政治分析师 · ⚖️ 国际法专家 · 💰 能源经济学家 · 🎖️ 军事战略家 · 🕊️ 冲突调解者 |
| 商业战略 | 📈 市场战略师 · 💼 运营负责人 · 🔍 竞争情报专家 · 💡 创新总监 · 📊 财务分析师 |

### 用法

```
[myproject] ❯ /brainstorm
  How many agents? (2-100, default 5) > 5

[myproject] ❯ /brainstorm improve plugin architecture
  How many agents? (2-100, default 5) > 3

[myproject] ❯ /brainstorm US-Iran geopolitics
  How many agents? (2-100, default 5) > 7
```

### 示例输出

```
[myproject] ❯ /brainstorm medical research funding
  How many agents? (2-100, default 5) > 3
Generating 3 topic-appropriate expert personas...
Starting 3-Agent Brainstorming Session on: medical research funding
Generating diverse perspectives...
🩺 Clinical Trials Director is thinking...
  └─ Perspective captured.
⚖️ Medical Ethics Committee Member is thinking...
  └─ Perspective captured.
💰 Health Economics Policy Analyst is thinking...
  └─ Perspective captured.
✓  Brainstorming complete! Results saved to brainstorm_outputs/brainstorm_20260405_224117.md

   ── Analysis from Main Agent ──
[synthesized Master Plan streams here…]
```

### 说明

- Brainstorm 使用**当前选中的模型**（用 `/model` 查看）。能力更强的模型（Claude Sonnet/Opus、GPT-4o 或大型本地模型）会得到更好的结果。
- 当 agent 数量较多（20+）时，根据模型速度不同，整个过程可能需要几分钟。
- 安装 `faker`（`pip install faker`）可生成随机人格名字；否则会回退到内置名字。
- 输出文件会积累在 `brainstorm_outputs/` 中 —— v3.05.5 已将其加入 `.gitignore`。
- 如果在 SSH 中输出混乱（重复行），可运行 `/config rich_live=false` 关闭 Rich Live 流式渲染。

---

## 主动式后台监控

Nano Claude Code v3.05.2 新增了一个**哨兵守护线程**，会在用户一段时间不活动后自动唤醒代理 —— 无需用户再次输入。这适用于持续监控日志、轮询市场脚本或定期检查代码等场景。

### 快速开始

```
[myproject] ❯ /proactive 5m
Proactive background polling: ON  (triggering every 300s of inactivity)

[myproject] ❯ keep monitoring the build log and alert me if errors appear

╭─ Claude ● ─────────────────────────
│ Understood. I'll check the build log each time I wake up.

[Background Event Triggered]
╭─ Claude ● ─────────────────────────
│ ⚙ Bash(tail -50 build.log)
│ ✓ → Build failed: ImportError in auth.py line 42
│ **Action needed:** fix the import before the next CI run.
```

### 命令

| 命令 | 说明 |
|---|---|
| `/proactive` | 显示当前状态（开/关和间隔） |
| `/proactive 5m` | 开启 —— 每 5 分钟无操作后触发 |
| `/proactive 30s` | 开启 —— 每 30 秒无操作后触发 |
| `/proactive 1h` | 开启 —— 每 1 小时无操作后触发 |
| `/proactive off` | 关闭哨兵轮询 |

时间后缀：`s` = 秒，`m` = 分钟，`h` = 小时。纯整数默认表示秒。

### 工作原理

- REPL 启动时会创建一个后台守护线程（默认暂停）。
- 守护线程每秒检查一次距离上次用户或代理交互已经过去多久。
- 当达到设定的无活动阈值时，它会向代理注入一个唤醒提示。
- 主代理循环使用的 `threading.Lock` 能确保唤醒不会打断当前会话 —— 它们会排队，并在当前轮完成后触发。
- 监视器异常会通过 `traceback` 记录，便于发现与调试。

### 与 SleepTimer 的互补关系

| | `SleepTimer` | `/proactive` |
|---|---|---|
| 谁发起 | 代理 | 用户 |
| 触发条件 | 从现在起固定延迟后 | 用户无活动 N 秒后 |
| 适用场景 | “10 分钟后回来检查” | “只要我不说话，你就持续监控” |

---

## 上下文压缩

长对话会自动压缩，以保持在模型的上下文窗口范围内。

**两层机制：**

1. **Snip** —— 几轮之后，旧工具输出（文件读取、bash 结果）会被截断。快速、无 API 成本。
2. **Auto-compact** —— 当 token 使用量超过上下文限制的 70% 时，旧消息会被模型总结成简短摘要。

这一过程是透明进行的，你无需手动操作。

---

## Diff 视图

当模型编辑或覆盖文件时，你会看到 git 风格 diff：

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

绿色行 = 新增，红色行 = 删除。新文件创建时则会显示摘要。

---

## CLAUDE.md 支持

在你的项目中放置一个 `CLAUDE.md` 文件，为模型提供关于代码库的持久上下文。Nano Claude 会自动发现它并将其注入系统提示中。

```
~/.claude/CLAUDE.md          # 全局 —— 作用于所有项目
/your/project/CLAUDE.md      # 项目级 —— 从 cwd 向上查找
```

**`CLAUDE.md` 示例：**

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

## 会话管理

### 存储布局

每次退出时会自动保存到三个位置：

```
~/.nano_claude/sessions/
├── history.json                          ← 主历史：所有会话（有上限）
├── mr_sessions/
│   └── session_latest.json              ← 始终是最近一次会话（用于 /resume）
└── daily/
    ├── 2026-04-05/
    │   ├── session_110523_a3f9.json     ← 每日会话文件，保留最新若干
    │   └── session_143022_b7c1.json
    └── 2026-04-04/
        └── session_183100_3b4c.json
```

每个会话文件都包含以下元数据：

```json
{
  "session_id": "a3f9c1b2",
  "saved_at": "2026-04-05 11:05:23",
  "turn_count": 8,
  "messages": [...]
}
```

### 退出时自动保存

每次通过 `/exit`、`/quit`、`Ctrl+C` 或 `Ctrl+D` 退出时，会话都会自动保存：

```
✓ Session saved → /home/.../.nano_claude/sessions/mr_sessions/session_latest.json
✓              → /home/.../.nano_claude/sessions/daily/2026-04-05/session_110523_a3f9.json  (id: a3f9c1b2)
✓   history.json: 12 sessions / 87 total turns
```

### 快速恢复

想继续上次会话时：

```bash
nano_claude
[myproject] ❯ /resume
✓  Session loaded from …/mr_sessions/session_latest.json (42 messages)
```

恢复指定文件：

```bash
/resume session_latest.json          # 从 mr_sessions/ 加载
/resume /absolute/path/to/file.json  # 从绝对路径加载
```

### 手动保存 / 加载

```bash
/save                          # 自动命名保存（session_TIMESTAMP_ID.json）
/save debug_auth_bug           # 保存为指定文件名，存入 ~/.nano_claude/sessions/

/load                          # 按日期分组的交互式列表
/load debug_auth_bug           # 按文件名加载
```

**`/load` 交互式列表示例：**

```
  ── 2026-04-05 ──
  [ 1] 11:05:23  id:a3f9c1b2  turns:8   session_110523_a3f9.json
  [ 2] 09:22:01  id:7e2d4f91  turns:3   session_092201_7e2d.json

  ── 2026-04-04 ──
  [ 3] 22:18:00  id:3b4c5d6e  turns:15  session_221800_3b4c.json

  ── Complete History ──
  [ H] Load ALL history  (3 sessions / 26 total turns)  /home/.../.nano_claude/sessions/history.json

  Enter number(s) (e.g. 1 or 1,2,3), H for full history, or Enter to cancel >
```

- 输入单个数字加载一个会话
- 输入逗号分隔数字（例如 `1,3`）按顺序合并多个会话
- 输入 `H` 加载完整历史 —— 会在确认前显示消息数和 token 估计

### 可配置上限

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `session_daily_limit` | `5` | `daily/` 中每天最多保留多少个会话文件 |
| `session_history_limit` | `100` | `history.json` 中最多保留多少个会话 |

```bash
/config session_daily_limit=10
/config session_history_limit=200
```

### history.json —— 完整对话历史

`history.json` 会把所有会话集中累积在一个地方，方便搜索完整历史记录或分析使用模式：

```json
{
  "total_turns": 150,
  "sessions": [
    {"session_id": "a3f9c1b2", "saved_at": "2026-04-05 11:05:23", "turn_count": 8, "messages": [...]},
    {"session_id": "7e2d4f91", "saved_at": "2026-04-05 09:22:01", "turn_count": 3, "messages": [...]}
  ]
}
```

---

## 云同步（GitHub Gist）

Nano Claude Code v3.05.3 新增了一个可选的会话云备份功能，基于 **GitHub Gist**。会话会以私有 Gist（JSON）形式保存，并可在 GitHub UI 中浏览。无额外依赖 —— 使用的是 Python 标准库 `urllib`。

### 设置（一次性）

1. 前往 [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. 启用 **`gist`** 权限
3. 复制 token，然后运行：

```
[myproject] ❯ /cloudsave setup ghp_xxxxxxxxxxxxxxxxxxxx
✓ GitHub token saved (logged in as: Chauncygu). Cloud sync is ready.
```

### 上传会话

```
[myproject] ❯ /cloudsave
Uploading session to GitHub Gist…
✓ Session uploaded → https://gist.github.com/abc123def456
```

附加可选描述：

```
[myproject] ❯ /cloudsave push auth refactor debug session
```

### 退出时自动同步

```
[myproject] ❯ /cloudsave auto on
✓ Auto cloud-sync ON — session will be uploaded to Gist on /exit.
```

此后，每次 `/exit` 或 `/quit` 前都会自动上传会话。

### 浏览与恢复

```
[myproject] ❯ /cloudsave list
  Found 3 session(s):
  abc123de…  2026-04-05 11:02  auth refactor debug session
  7f9e12ab…  2026-04-04 22:18  proactive monitoring test
  3b4c5d6e…  2026-04-04 18:31

[myproject] ❯ /cloudsave load abc123de...full-gist-id...
✓ Session loaded from Gist (42 messages).
```

### 命令参考

| 命令 | 说明 |
|---|---|
| `/cloudsave setup <token>` | 保存 GitHub token（需要 `gist` 权限） |
| `/cloudsave` | 将当前会话上传到新的或已有的 Gist |
| `/cloudsave push [desc]` | 上传时附带可选描述 |
| `/cloudsave auto on\|off` | 开关退出时自动上传 |
| `/cloudsave list` | 列出所有 nano-claude-code Gists |
| `/cloudsave load <gist_id>` | 下载并恢复一个会话 |

---

## 项目结构

```
nano_claude_code/
├── nano_claude.py        # 入口：REPL + slash commands + diff 渲染 + Rich Live 流式显示 + 主动式哨兵守护线程
├── agent.py              # Agent 循环：streaming、工具调度、压缩
├── providers.py          # 多 provider：Anthropic、OpenAI-compatible streaming
├── tools.py              # 核心工具（Read/Write/Edit/Bash/Glob/Grep/Web/NotebookEdit/GetDiagnostics）+ registry 连接
├── tool_registry.py      # 工具插件注册表：注册、查找、执行
├── compaction.py         # 上下文压缩：snip + 自动摘要
├── context.py            # 系统提示构建器：CLAUDE.md + git + memory
├── config.py             # 配置加载/保存/默认值；DAILY_DIR、SESSION_HIST_FILE 路径
├── cloudsave.py          # GitHub Gist 云同步（上传/下载/列出会话）
│
├── multi_agent/          # 多代理包
│   ├── __init__.py       # Re-exports
│   ├── subagent.py       # AgentDefinition、SubAgentManager、worktree 辅助函数
│   └── tools.py          # Agent、SendMessage、CheckAgentResult、ListAgentTasks、ListAgentTypes
├── subagent.py           # 向后兼容 shim → multi_agent/
│
├── memory/               # 记忆包
│   ├── __init__.py       # Re-exports
│   ├── types.py          # MEMORY_TYPES 与格式指南
│   ├── store.py          # 保存/加载/删除/搜索、重建 MEMORY.md 索引
│   ├── scan.py           # MemoryHeader、年龄/新鲜度辅助函数
│   ├── context.py        # get_memory_context()、截断、AI search
│   └── tools.py          # MemorySave、MemoryDelete、MemorySearch、MemoryList
├── memory.py             # 向后兼容 shim → memory/
│
├── skill/                # 技能包
│   ├── __init__.py       # Re-exports；导入 builtin 以注册内置技能
│   ├── loader.py         # SkillDef、解析、load_skills、find_skill、substitute_arguments
│   ├── builtin.py        # 内置技能：/commit、/review
│   ├── executor.py       # execute_skill()：inline 或 forked 子代理
│   └── tools.py          # Skill、SkillList
├── skills.py             # 向后兼容 shim → skill/
│
├── mcp/                  # MCP（Model Context Protocol）包
│   ├── __init__.py       # Re-exports
│   ├── types.py          # MCPServerConfig、MCPTool、MCPServerState、JSON-RPC 辅助函数
│   ├── client.py         # StdioTransport、HttpTransport、MCPClient、MCPManager
│   ├── config.py         # 加载 .mcp.json（项目）+ ~/.nano_claude/mcp.json（用户）
│   └── tools.py          # 自动发现 + 将 MCP 工具注册到 tool_registry
│
├── voice/                # 语音输入包（v3.05）
│   ├── __init__.py       # 公共 API：check_voice_deps、voice_input
│   ├── recorder.py       # 音频捕获：sounddevice → arecord → sox rec
│   ├── stt.py            # STT：faster-whisper → openai-whisper → OpenAI API
│   └── keyterms.py       # 从 git 分支 + 项目文件中抽取代码领域词汇
│
└── tests/                # 239+ 单元测试
    ├── test_mcp.py
    ├── test_memory.py
    ├── test_skills.py
    ├── test_subagent.py
    ├── test_tool_registry.py
    ├── test_compaction.py
    ├── test_diff_view.py
    └── test_voice.py      # 29 个语音测试（无需硬件）
```

> **给开发者：** 每个特性包（`multi_agent/`、`memory/`、`skill/`、`mcp/`、`voice/`）都是自包含的。添加自定义工具时，只需在任意被 `tools.py` 导入的模块中调用 `register_tool(ToolDef(...))`。

---

## FAQ

**Q：如何添加一个 MCP server？**

方式 1 —— 通过 REPL（stdio server）：
```
/mcp add git uvx mcp-server-git
```

方式 2 —— 在项目中创建 `.mcp.json`：
```json
{
  "mcpServers": {
    "git": {"type": "stdio", "command": "uvx", "args": ["mcp-server-git"]}
  }
}
```

然后运行 `/mcp reload` 或重启。使用 `/mcp` 查看连接状态。

**Q：某个 MCP server 报错了，怎么调试？**

```
/mcp                    # 显示每个 server 的错误信息
/mcp reload git         # 尝试重连
```

如果 server 使用 stdio，请确认命令在你的 `$PATH` 中：
```bash
which uvx               # 应输出路径
uvx mcp-server-git      # 手动运行查看错误
```

**Q：可以使用需要认证的 MCP server 吗？**

对于带 Bearer token 的 HTTP/SSE server：
```json
{
  "mcpServers": {
    "my-api": {
      "type": "sse",
      "url": "https://myserver.example.com/sse",
      "headers": {"Authorization": "Bearer sk-my-token"}
    }
  }
}
```

对于使用环境变量认证的 stdio server：
```json
{
  "mcpServers": {
    "brave": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-server-brave-search"],
      "env": {"BRAVE_API_KEY": "your-key"}
    }
  }
}
```

**Q：我本地的 Ollama 模型无法进行工具调用。**

并非所有模型都支持 function calling。请使用推荐的可工具调用模型之一：`qwen2.5-coder`、`llama3.3`、`mistral` 或 `phi4`。

```bash
ollama pull qwen2.5-coder
nano_claude --model ollama/qwen2.5-coder
```

**Q：如何连接运行 vLLM 的远程 GPU 服务器？**

```
/config custom_base_url=http://your-server-ip:8000/v1
/config custom_api_key=your-token
/model custom/your-model-name
```

**Q：如何查看 API 成本？**

```
/cost

  Input tokens:  3,421
  Output tokens:   892
  Est. cost:     $0.0648 USD
```

**Q：同一会话里能否使用多个 API Key？**

可以。先通过环境变量或 `/config` 设置好所有需要的 Key，然后自由切换模型 —— 每次调用都会使用当前 active provider 对应的 Key。

**Q：如何让某个模型在所有项目中都可用？**

把 Key 加入 `~/.bashrc` 或 `~/.zshrc`。在 `~/.nano_claude/config.json` 中设置默认模型：

```json
{ "model": "claude-sonnet-4-6" }
```

**Q：Qwen / Zhipu 返回乱码。**

请确认你的 `DASHSCOPE_API_KEY` / `ZHIPU_API_KEY` 正确，且账号有足够额度。这两个 provider 都使用 UTF-8，并且对中文支持良好。

**Q：可以把输入通过管道传给 nano claude 吗？**

```bash
echo "Explain this file" | nano_claude --print --accept-all
cat error.log | nano_claude -p "What is causing this error?"
```

**Q：如何把它作为 CLI 工具在任意目录运行？**

使用 `uv tool install` —— 它会创建隔离环境并把 `nano_claude` 加入 PATH：

```bash
cd nano-claude-code
uv tool install .
```

之后就可以在任意目录直接运行 `nano_claude`。如果拉取了更新，使用 `uv tool install . --reinstall` 重新安装即可。

**Q：如何配置语音输入？**

```bash
# 最小配置（本地、离线、无需 API key）：
pip install sounddevice faster-whisper numpy

# 然后在 REPL 中：
/voice status          # 验证后端是否被检测到
/voice                 # 直接说出你的 prompt
```

首次使用时，`faster-whisper` 会自动下载 `base` 模型（约 150 MB）。  
如需更高准确率，可使用更大的模型：`export NANO_CLAUDE_WHISPER_MODEL=small`

**Q：语音输入转写不准（尤其漏掉代码术语）。**

关键词增强器已经会从你的 git 分支和项目文件中注入编程领域词汇。  
对于长期使用的领域术语，可以把它们写入 `.nano_claude/voice_keyterms.txt`（每行一个词）—— 每次录音前都会自动读取。

**Q：语音输入支持中文 / 日语 / 其他语言吗？**

支持。录音前先设置语言：

```
/voice lang zh    # 普通话
/voice lang ja    # 日语
/voice lang auto  # 恢复自动检测（默认）
```

Whisper 支持 99 种语言。`auto` 自动检测通常已经很好，但对于较短语音，显式指定语言会更准确。
