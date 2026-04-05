# Keep Memory Provider

Reflective memory via the `keep-skill` daemon. Keep adds semantic search, versioned session capture, tagging, and prompt-driven reflection on top of Hermes's built-in memory.

## Requirements

- `uv pip install keep-skill`
- Run `hermes memory setup` and select `keep`

## Setup

```bash
uv pip install keep-skill
hermes memory setup    # select "keep", choose providers
```

Always use `hermes memory setup` — it bootstraps the Keep store config (embedding/summarization providers) and writes `KEEP_STORE_PATH` to `.env`. Setting `memory.provider` manually without running setup leaves the store uninitialized.

## Store location

The setup wizard creates the store at `$HERMES_HOME/keep` (e.g. `~/.hermes/keep`). If `KEEP_STORE_PATH` is set in the environment, `initialize()` uses that path instead.

## Tools

| Tool | Description |
|------|-------------|
| `keep_flow` | Full Keep state-doc flow surface: search, get, put, tag, move, stats, and more |
| `keep_prompt` | Render Keep agent prompts like `reflect`, `session-start`, and `conversation` |
| `keep_help` | Read Keep help topics and operation docs |

For `keep_flow`, prefer the normal `state + params` form in prompts, docs, and examples:

```text
keep_flow(state="get", params={"id": "now"})
keep_flow(state="list", params={"prefix": ".library", "include_hidden": true})
keep_flow(state="query-resolve", params={"query": "recent work"})
```

`state_doc_yaml` exists for advanced custom inline flows, but it should not be the default example shape for Hermes agents. Lighter models are noticeably more reliable with `state + params`.
