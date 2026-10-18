# Phase 00 — Baseline Stabilization and MCP Client Connection

## Objective

Make the current MCP server stable enough to be launched by Claude Code, Codex, VS Code Copilot, and PyCharm AI before adding advanced retrieval features.

This phase must prove:

```text
launcher scripts work
MCP server starts
skills repo is reachable
indexed knowledge profile is selected correctly
Chroma collections can be built/reused
MCP client can call tools
```

Do not add Qwen, BM25, reranking, or incremental indexing in this phase.

---

## Current components involved

```text
src/mcp_server/mcp_server.py
config.yaml
.env
~/.local/bin/personal-mcp-indexed
~/.local/bin/personal-mcp-full
GitLab skills repo
storage/chroma/
```

---

## Desired architecture after this phase

```text
Claude Code / Codex / VS Code / PyCharm
        ↓ launches
~/.local/bin/personal-mcp-indexed
        ↓ sets env var
MCP_KNOWLEDGE_PROFILE=indexed
        ↓ starts
python -m mcp_server
        ↓ exposes tools
collection_status, knowledge_profile_status, retrieve_skills, retrieve_knowledge, etc.
```

---

## Step-by-step implementation

### Step 1 — Validate launcher scripts

Run:

```bash
ls -l "$HOME/.local/bin" | grep personal-mcp
which personal-mcp-indexed
which personal-mcp-full
cat "$HOME/.local/bin/personal-mcp-indexed"
cat "$HOME/.local/bin/personal-mcp-full"
```

Expected indexed launcher:

```bash
#!/usr/bin/env bash
cd "/Users/oscarbonnet/PycharmProjects/mcp-server"
export MCP_KNOWLEDGE_PROFILE=indexed
exec "/Users/oscarbonnet/PycharmProjects/mcp-server/.venv/bin/python" -m mcp_server
```

Expected full launcher:

```bash
#!/usr/bin/env bash
cd "/Users/oscarbonnet/PycharmProjects/mcp-server"
export MCP_KNOWLEDGE_PROFILE=full
exec "/Users/oscarbonnet/PycharmProjects/mcp-server/.venv/bin/python" -m mcp_server
```

### Step 2 — Validate Python syntax

Run:

```bash
cd "/Users/oscarbonnet/PycharmProjects/mcp-server"
poetry run python -m py_compile src/mcp_server/mcp_server.py
```

Expected output: no output.

### Step 3 — Validate knowledge profile config

Run:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import KNOWLEDGE_PROFILE, _get_knowledge_paths
print(KNOWLEDGE_PROFILE)
print(_get_knowledge_paths())
PY
```

Expected:

```text
indexed
['/Users/oscarbonnet/Documents/80–89 Knowledge Base/87 Indexed']
```

### Step 4 — Build indexed collections once

Run:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import _rebuild_all_indexes
print(_rebuild_all_indexes(force_rebuild=True))
PY
```

Expected output includes:

```text
knowledge_collection: knowledge_indexed
knowledge_profile: indexed
mode: force_rebuild
```

### Step 5 — Confirm collection reuse

Run:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import _rebuild_all_indexes, collection_status
print(_rebuild_all_indexes(force_rebuild=False))
print(collection_status())
PY
```

Expected logs include:

```text
Using existing Chroma collection 'knowledge_indexed' with X vectors.
```

### Step 6 — Start server manually

Run:

```bash
personal-mcp-indexed
```

Expected: server starts and waits.

Do not type into the terminal. Stop with `Ctrl+C`.

### Step 7 — Connect Claude Code

Run:

```bash
claude mcp add --scope user personal-assistant-indexed -- \
  "/Users/oscarbonnet/.local/bin/personal-mcp-indexed"
```

Then:

```bash
claude mcp list
```

Expected: `personal-assistant-indexed` appears.

### Step 8 — MCP client test prompts

Ask Claude Code:

```text
Use the personal-assistant-indexed MCP server to show knowledge_profile_status.
```

Expected:

```text
active_profile = indexed
paths include /87 Indexed
```

Then ask:

```text
Use the personal-assistant-indexed MCP server to show collection_status.
```

Expected:

```text
skills vectors > 0
knowledge collection = knowledge_indexed
memory collection exists, possibly 0 vectors if no memory files yet
```

Then ask:

```text
Use the personal-assistant-indexed MCP server to retrieve skills for reviewing an MCP architecture.
```

Expected: result includes `skills/active/architecture_review.md`.

---

## Desired outputs

By the end of this phase:

```text
- indexed launcher works
- full launcher exists but does not need to be used yet
- MCP client can call diagnostic tools
- retrieve_skills works
- retrieve_knowledge works if /87 Indexed has files
- no advanced features added yet
```

---

## Acceptance criteria

Phase is complete if:

```text
[ ] `personal-mcp-indexed` starts server.
[ ] `collection_status()` works through MCP client.
[ ] `knowledge_profile_status()` works through MCP client.
[ ] `retrieve_skills("MCP architecture")` returns architecture skill.
[ ] `retrieve_knowledge()` returns results from /87 Indexed if files exist.
[ ] Server logs do not pollute stdout.
```

---

## Common failure modes

### Server starts but MCP client cannot connect

Check launcher path and executable permissions:

```bash
chmod +x ~/.local/bin/personal-mcp-indexed
```

### Retrieval returns placeholder document

Likely cause: collection was never built with `force_rebuild=True`.

Fix:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run python - <<'PY'
from mcp_server.mcp_server import _rebuild_all_indexes
print(_rebuild_all_indexes(force_rebuild=True))
PY
```

### GitLab skills not found

Check:

```text
config.yaml GitLab project ID
GITLAB_TOKEN in .env
skills_path = skills/active
skills repo has .md files
```
