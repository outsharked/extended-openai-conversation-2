# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Extended OpenAI Conversation is a Home Assistant custom component that extends the built-in OpenAI Conversation integration. It adds a skills system, template functions, bash execution, file operations, AI Task entities, vision support, and multi-provider support (OpenAI, Azure OpenAI, compatible servers).

## Commands

### Lint & Format
```bash
ruff check custom_components/
ruff format --check custom_components/
# Auto-fix:
ruff check --fix custom_components/
ruff format custom_components/
```

### Type Check
```bash
mypy custom_components/extended_openai_conversation
```

### Tests
```bash
# All tests with coverage
pytest tests/ -v --timeout=30 --cov=custom_components/extended_openai_conversation --cov-report=term-missing

# Single test file
pytest tests/functions/test_native.py -v

# Single test
pytest tests/functions/test_native.py::test_function_name -v
```

### Dev Setup
```bash
pip install -e ".[dev]"
pip install -r requirements_test.txt
```

CI dynamically fetches dependency versions from HA core's dev branch (openai, hassil, beautifulsoup4, etc.) and tests against both stable and dev Home Assistant.

## Architecture

All source lives under `custom_components/extended_openai_conversation/`.

### Entry Point & Platforms
`__init__.py` sets up the OpenAI API client (`AsyncClient` or `AsyncAzureOpenAI`) and registers two platforms:
- **conversation** (`conversation.py`, `entity.py`) — `ExtendedOpenAIAgentEntity` handles user messages via `async_process()`: builds system prompt with exposed entities + skills, calls OpenAI API with tools, executes tool calls through Tools, iterates until completion.
- **ai_task** (`ai_task.py`) — `ExtendedOpenAITaskEntity` handles background AI tasks with structured output (JSON schema), no tool calling.

### Functions (`functions/`)
Abstract `Function` base class (`functions/base.py`) with implementations in separate files: `native` (HA services), `template` (Jinja2), `script` (HA scripts), `web` (REST/scrape), `bash`, `file` (read/write/edit), `sqlite`, `composite` (chains multiple). Each is registered by type string and resolved at tool-call time.

### Skills System (`skills.py`)
`SkillManager` (singleton) discovers and loads skills from `config/extended_openai_conversation/skills/`. Each skill is a directory with a `SKILL.md` file (YAML frontmatter + markdown content). `SkillMdParser` handles parsing. Skills are lazy-loaded — only metadata is kept in memory until content is requested.

### Config Flow (`config_flow.py`)
`ConfigFlow` for initial setup (API key, base URL, provider). `ConfigSubentryFlow` for per-conversation and per-ai-task settings (model, tokens, temperature, prompt template, functions YAML, skills selection).

### Services (`services.py`)
`query_image` (vision API), `change_config` (runtime config updates), `reload_skills`, `download_skill` (from GitHub).

### Security
- Workspace restriction for file operations (default: `extended_openai_conversation/`)
- Shell command deny patterns (rm -rf, format, etc.)
- File size limits (1 MB), bash timeout (300s), shell output limit (10,000 chars)

## Test Structure

Tests use `pytest` with `pytest-asyncio` and `pytest-homeassistant-custom-component`. All tests are async (`asyncio_mode = "auto"` in pyproject.toml).

```
tests/
├── conftest.py              # Fixtures: hass mock, llm_context, exposed_entities, temp_db_path
├── helpers.py               # Test utilities
├── fixtures/                # Test data (functions YAML, mock responses)
└── functions/               # One test file per function type
```

## Key Conventions

- Python 3.14+, HA minimum 2026.2.0b0
- Ruff rules: E, W, F, I, UP, RUF, B, SIM (E501 and B008 ignored)
- Import aliases enforced: `voluptuous` → `vol`, `homeassistant.helpers.config_validation` → `cv`
- All I/O is async; use `async_add_executor_job()` for sync code
- Custom exceptions in `exceptions.py` (EntityNotFound, CallServiceError, TokenLengthExceededError, etc.)
- System prompt uses Jinja2 templates with context: `ha_name`, `exposed_entities`, `current_device_id`, `user_input`, `skills`

## Release & Commit Conventions

This is a public repo with real installs via HACS — commits and release notes are user-facing artifacts, not dev notes.

- **Never put a private Claude session URL in a commit message on this repo.** A `Claude-Session: https://claude.ai/code/session_...` trailer is meaningless (and confusing) to anyone else, and it doesn't stay contained to dev process — GitHub's own UI surfaces the target commit's message in several places (the Tags list in particular, which falls back to showing the underlying commit when a tag has no separate annotated message), so a private link baked into a commit ends up effectively public. If attributing AI assistance in a commit at all, use the same generic, public-safe line PR descriptions already use: `🤖 Generated with [Claude Code](https://claude.com/claude-code)` — no session link, no `Co-Authored-By` block front-and-center. Keep it a brief trailing mention, never the first thing a reader sees.
- **GitHub Release notes must lead with the actual user-facing content** — features and fixes, in plain language, referencing issue/PR numbers — covering everything since the *previous* release, not just the most recent commit's diff. Don't rely on `gh release create` without `--notes`, and don't let the release's summary just restate one commit's message: write it as if a user who never saw the commit history needs to understand what changed and why it matters to them.
- When cutting a tag, prefer writing real `--notes` over `--generate-notes` for exactly this reason — auto-generated notes tend to surface raw commit/PR titles rather than a synthesized "what changed for users" summary.
