from __future__ import annotations

from mcp_server.retrieval.token_budget import (
    TokenBudget,
    check_token_budget,
    estimate_tokens,
    new_run_id,
    record_token_ledger,
    token_ledger_summary,
)


def test_estimate_tokens_is_deterministic():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("") == 1


def test_token_ledger_records_estimated_usage(tmp_path):
    db_path = tmp_path / "manifest.sqlite"
    run_id = new_run_id()

    record_token_ledger(
        db_path,
        run_id=run_id,
        provider="voyage",
        model="voyage-code-3",
        usage_type="document_embedding_estimate",
        tokens=123,
        estimated_or_actual="estimated",
        notes="test",
    )

    summary = token_ledger_summary(db_path)

    assert summary["total_tokens_all"] == 123
    assert summary["total_tokens_actual"] == 0
    assert summary["total_tokens_estimated"] == 123
    assert summary["by_model"][0]["provider"] == "voyage"
    assert summary["by_model"][0]["model"] == "voyage-code-3"
    assert summary["by_model"][0]["tokens"] == 123


def test_token_budget_hard_stop_blocks_run():
    status = check_token_budget(
        estimated_tokens=50,
        historical_tokens=175,
        budget=TokenBudget(
            max_tokens_per_run=100,
            warning_threshold=150,
            hard_stop_threshold=200,
        ),
    )

    assert status["exceeds_warning_threshold"] is True
    assert status["exceeds_hard_stop_threshold"] is True
    assert status["allowed"] is False


def test_repeated_dry_run_estimates_do_not_count_as_actual_spend(tmp_path):
    db_path = tmp_path / "manifest.sqlite"

    for _ in range(3):
        record_token_ledger(
            db_path,
            run_id=new_run_id(),
            provider="voyage",
            model="voyage-code-3",
            usage_type="document_embedding_estimate",
            tokens=90,
            estimated_or_actual="estimated",
        )

    summary = token_ledger_summary(db_path)
    status = check_token_budget(
        estimated_tokens=20,
        historical_tokens=summary["total_tokens_actual"],
        budget=TokenBudget(
            max_tokens_per_run=100,
            warning_threshold=100,
            hard_stop_threshold=100,
        ),
    )

    assert summary["total_tokens_all"] == 270
    assert summary["total_tokens_estimated"] == 270
    assert summary["total_tokens_actual"] == 0
    assert status["projected_total_tokens"] == 20
    assert status["exceeds_hard_stop_threshold"] is False
    assert status["allowed"] is True
