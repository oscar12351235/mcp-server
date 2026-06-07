# Personal MCP Server

Local MCP server for personal skills, local knowledge retrieval, project file access, and persistent memory proposals.

This server is designed to be used by Claude Code, VS Code Copilot, PyCharm AI Assistant, or any MCP-compatible client.

## Purpose

This project provides a local MCP server that exposes tools for:

* Retrieving reusable skills stored in a GitLab skills repository
* Retrieving local knowledge from selected folders
* Retrieving personal/project memory from local markdown files
* Reading allowlisted local files
* Listing allowlisted local directories
* Reading allowlisted GitLab files
* Proposing new memory entries
* Proposing new GitLab skills through merge requests

The LLM is not provided by this project. Claude, Copilot, or another MCP client acts as the LLM and orchestrator.

## Architecture

```text
                       Claude / Copilot / PyCharm AI
                                     ↓
                              Local MCP server
                                     ↓
         ┌───────────────────────────┬───────────────────────────┐
         ↓                           ↓                           ↓
    GitLab skills repo        Local knowledge               Local memory
    skills/*.md               books/notes/docs              preferences/projects
         ↓                           ↓                           ↓
    LlamaIndex retrieval      LlamaIndex retrieval          LlamaIndex retrieval
         ↓             ↓             ↓              ↓            ↓
                       Local ChromaDB vector storage
```

## Main components

* `src/mcp_server/mcp_server.py`: main MCP server implementation
* `config.yaml`: local configuration, paths, retrieval settings, and security allowlists
* `.env`: local secrets such as GitLab token; not committed
* `storage/chroma/`: local ChromaDB vector storage; not committed
* GitLab `skills` repo: reusable skill markdown files

## Knowledge profiles

The server supports multiple knowledge profiles.

Typical profiles:

```yaml
knowledge:
  default_profile: "indexed"

  profiles:
    indexed:
      paths:
        - "/Users/oscarbonnet/Documents/80–89 Knowledge Base/87 Indexed"

    full:
      paths:
        - "/Users/oscarbonnet/Documents/80–89 Knowledge Base"
```

Use `indexed` for daily work.

Use `full` only when you explicitly want to search the entire knowledge base, including raw PDFs/books.

## Environment variables

Required:

```env
GITLAB_TOKEN=your_gitlab_token
GITLAB_BASE_URL=https://gitlab.com/api/v4
```

Optional:

```env
MCP_KNOWLEDGE_PROFILE=indexed
```

Supported values depend on `config.yaml`, usually:

* `indexed`
* `full`

## Running locally

From the project root:

```bash
poetry run personal-mcp
```

Or:

```bash
poetry run python -m mcp_server
```

For indexed mode:

```bash
MCP_KNOWLEDGE_PROFILE=indexed poetry run personal-mcp
```

For full mode:

```bash
MCP_KNOWLEDGE_PROFILE=full poetry run personal-mcp
```

## Recommended launcher scripts

Daily mode:

```bash
personal-mcp-indexed
```

Full-library mode:

```bash
personal-mcp-full
```

These scripts should live in:

```text
/Users/oscarbonnet/.local/bin
```

They make it easier for MCP clients to launch the server with the right environment variables.

## MCP tools

The server exposes these tools:

### `refresh_indexes`

Rebuilds skills, knowledge, and memory indexes.

Use after adding or modifying skills, indexed knowledge files, or memory files.

### `index_status`

Shows current index status.

### `knowledge_profile_status`

Shows the active knowledge profile and the paths used for retrieval.

### `retrieve_skills`

Retrieves relevant skill files from the GitLab skills repo.

Example use:

```text
Retrieve skills for reviewing an MCP architecture.
```

### `retrieve_knowledge`

Retrieves relevant local knowledge from the active knowledge profile.

Example use:

```text
Retrieve knowledge about PCA in markets.
```

### `retrieve_memory`

Retrieves relevant personal/project memory.

Example use:

```text
Retrieve memory about my coding preferences.
```

### `read_local_file`

Reads one allowlisted local file.

### `list_local_files`

Lists files inside an allowlisted local directory.

### `get_gitlab_file`

Reads one allowlisted GitLab file.

### `list_gitlab_repo`

Lists an allowlisted GitLab repository tree.

### `propose_memory_update`

Creates a memory proposal in the local memory proposal folder.

This does not directly update permanent memory.

### `propose_new_skill`

Creates a proposed skill in the GitLab skills repo using a branch and merge request.

This does not push directly to `main`.

## Security model

The server uses allowlisted read and write roots.

Recommended principles:

* Do not allow full home-directory access
* Do not allow direct writes to secrets
* Do not allow direct writes to the skills repo `main` branch
* Store memory changes as proposals first
* Keep `.env`, `.venv`, and `storage/` out of Git

Important blocked paths should include:

```yaml
never_read:
  - "/Users/oscarbonnet/.ssh"
  - "/Users/oscarbonnet/.gnupg"
  - "/Users/oscarbonnet/Library"
  - "/Users/oscarbonnet/Downloads"
  - "/Users/oscarbonnet/Desktop"

never_write:
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/.env"
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/.venv"
  - "/Users/oscarbonnet/PycharmProjects/mcp-server/storage"
  - "/Users/oscarbonnet/PycharmProjects/python-project-template"
```

## GitLab skills repo

The skills repo should contain:

```text
skills/
  active/
    architecture_review.md
    python_debugging.md
    refactor_planning.md
    test_design.md

  _proposals/
    .gitkeep

usage/
  usage_log.jsonl

registry.yaml
README.md
```

In `config.yaml`, the MCP server should point to the numeric GitLab project ID of the skills repo:

```yaml
gitlab:
  skills:
    project_id: "YOUR_NUMERIC_SKILLS_PROJECT_ID"
    branch: "main"
    skills_path: "skills/active"
    proposal_path: "skills/_proposals"
    write_mode: "merge_request"
```

## Local ChromaDB

The server uses local ChromaDB through `chromadb.PersistentClient`.

No Chroma Cloud account is required.

Vectors are stored locally in:

```text
storage/chroma/
```

This folder should not be committed.

## Recommended workflow

1. Run the server in indexed mode.
2. Ask the MCP client to call `knowledge_profile_status`.
3. Ask the MCP client to call `refresh_indexes`.
4. Use `retrieve_skills`, `retrieve_knowledge`, or `retrieve_memory`.
5. Add high-signal notes or summaries to the indexed knowledge folder.
6. Use full mode only when searching the entire raw knowledge base.

## Development notes

Run formatting:

```bash
poetry run black .
```

Run linting:

```bash
poetry run ruff check .
```

Run type checking:

```bash
poetry run mypy .
```

Run tests, if present:

```bash
poetry run pytest
```

## Current limitations

* Full knowledge mode can be slow because large PDFs create many document objects.
* PDF parsing quality depends on the PDF structure.
* `refresh_indexes()` currently rebuilds collections from scratch.
* Incremental indexing is not implemented yet.
* Spreadsheet, image, and table parsing should be handled separately before being added to the main RAG pipeline.

