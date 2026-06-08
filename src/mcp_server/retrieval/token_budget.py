from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.retrieval.manifest import init_manifest_db, utc_now


@dataclass(frozen=True)
class TokenBudget:
    max_tokens_per_run: int
    warning_threshold: int
    hard_stop_threshold: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def token_budget_from_config(config: dict[str, Any]) -> TokenBudget:
    voyage_config = config.get("voyage", {})
    return TokenBudget(
        max_tokens_per_run=int(voyage_config.get("max_tokens_per_run_default", 2_000_000)),
        warning_threshold=int(voyage_config.get("warn_at_total_tokens", 150_000_000)),
        hard_stop_threshold=int(voyage_config.get("hard_stop_at_total_tokens", 190_000_000)),
    )


def new_run_id() -> str:
    return str(uuid.uuid4())


def record_token_ledger(
    db_path: str | Path,
    *,
    run_id: str,
    provider: str,
    model: str,
    usage_type: str,
    tokens: int,
    estimated_or_actual: str,
    file_id: str | None = None,
    chunk_id: str | None = None,
    notes: str | None = None,
) -> None:
    init_manifest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO token_ledger (
                timestamp, run_id, provider, model, usage_type, tokens,
                estimated_or_actual, file_id, chunk_id, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                run_id,
                provider,
                model,
                usage_type,
                tokens,
                estimated_or_actual,
                file_id,
                chunk_id,
                notes,
            ),
        )


def token_ledger_summary(db_path: str | Path) -> dict[str, Any]:
    init_manifest_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT provider, model, estimated_or_actual, COALESCE(SUM(tokens), 0) AS tokens
            FROM token_ledger
            GROUP BY provider, model, estimated_or_actual
            ORDER BY provider, model, estimated_or_actual
            """
        ).fetchall()
        totals = conn.execute(
            """
            SELECT
                COALESCE(SUM(tokens), 0) AS total_all,
                COALESCE(SUM(CASE WHEN estimated_or_actual = 'actual' THEN tokens ELSE 0 END), 0)
                    AS total_actual,
                COALESCE(
                    SUM(CASE WHEN estimated_or_actual = 'estimated' THEN tokens ELSE 0 END),
                    0
                ) AS total_estimated
            FROM token_ledger
            """
        ).fetchone()

    return {
        "total_tokens_all": int(totals[0] or 0),
        "total_tokens_actual": int(totals[1] or 0),
        "total_tokens_estimated": int(totals[2] or 0),
        "by_model": [
            {
                "provider": row[0],
                "model": row[1],
                "estimated_or_actual": row[2],
                "tokens": int(row[3] or 0),
            }
            for row in rows
        ],
    }


def check_token_budget(
    *,
    estimated_tokens: int,
    historical_tokens: int,
    budget: TokenBudget,
) -> dict[str, Any]:
    projected_total = historical_tokens + estimated_tokens
    return {
        "estimated_tokens": estimated_tokens,
        "historical_tokens": historical_tokens,
        "projected_total_tokens": projected_total,
        "max_tokens_per_run": budget.max_tokens_per_run,
        "warning_threshold": budget.warning_threshold,
        "hard_stop_threshold": budget.hard_stop_threshold,
        "exceeds_run_limit": estimated_tokens > budget.max_tokens_per_run,
        "exceeds_warning_threshold": projected_total >= budget.warning_threshold,
        "exceeds_hard_stop_threshold": projected_total >= budget.hard_stop_threshold,
        "allowed": estimated_tokens <= budget.max_tokens_per_run
        and projected_total < budget.hard_stop_threshold,
    }
