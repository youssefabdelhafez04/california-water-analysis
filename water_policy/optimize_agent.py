"""Dev/test optimization protocol for the California water SQL agent."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from water_policy.chat_agent import WaterDataAgent, run_select
from water_policy.evaluate_agent import classify_failure, results_match


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
QUESTIONS = ROOT / "evaluation" / "questions.json"
DEV_QUESTIONS = ROOT / "evaluation" / "dev_questions.json"
TEST_QUESTIONS = ROOT / "evaluation" / "heldout_questions.json"
SPLIT_MANIFEST = RESULTS / "agent_eval_split.json"
DATABASE = RESULTS / "water_analysis.sqlite"
BASELINE_FULL = RESULTS / "agent_eval.csv"
DEV_DETAIL = RESULTS / "agent_dev_experiments.csv"
DEV_SUMMARY = RESULTS / "agent_dev_summary.csv"
FINAL_TEST = RESULTS / "agent_final_test.csv"
FINAL_TEST_PARTIAL = RESULTS / "agent_final_test.partial.csv"
HELDOUT_COMPARISON = RESULTS / "agent_heldout_comparison.csv"
FINAL_CONFIG = RESULTS / "agent_final_config.json"
README = ROOT / "README.md"
SPLIT_SEED = 20261009

# Standard short-context rates per 1M tokens, fetched from official OpenAI pricing
# on 2026-10-09: https://developers.openai.com/api/docs/pricing
PRICING = {
    "gpt-5-mini": {"input": 0.25, "cached": 0.025, "cache_write": 0.25, "output": 2.00},
    "gpt-6.1-sol": {"input": 2.00, "cached": 0.10, "cache_write": 2.50, "output": 10.00},
}

CONFIGS = [
    {
        "name": "baseline",
        "model": "gpt-5-mini",
        "use_schema_guide": False,
        "use_examples": False,
        "inspect_values_first": False,
        "service_tier": "default",
    },
    {
        "name": "schema_guide",
        "model": "gpt-5-mini",
        "use_schema_guide": True,
        "use_examples": False,
        "inspect_values_first": False,
        "service_tier": "default",
    },
    {
        "name": "schema_plus_examples",
        "model": "gpt-5-mini",
        "use_schema_guide": True,
        "use_examples": True,
        "inspect_values_first": False,
        "service_tier": "default",
    },
    {
        "name": "schema_examples_distinct_check",
        "model": "gpt-5-mini",
        "use_schema_guide": True,
        "use_examples": True,
        "inspect_values_first": True,
        "service_tier": "default",
    },
    {
        "name": "stronger_model",
        "model": "gpt-6.1-sol",
        "use_schema_guide": True,
        "use_examples": True,
        "inspect_values_first": True,
        "service_tier": "default",
    },
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare_split() -> dict[str, Any]:
    questions = json.loads(QUESTIONS.read_text())
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in questions:
        grouped.setdefault(item["type"], []).append(item)
    rng = random.Random(SPLIT_SEED)
    dev: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for question_type in sorted(grouped):
        items = sorted(grouped[question_type], key=lambda item: item["id"])
        rng.shuffle(items)
        dev.extend(items[:5])
        test.extend(items[5:])
    dev.sort(key=lambda item: item["id"])
    test.sort(key=lambda item: item["id"])
    manifest = {
        "seed": SPLIT_SEED,
        "method": "Within each question type, deterministic seeded shuffle; five dev and five held out.",
        "dev_ids": [item["id"] for item in dev],
        "heldout_ids": [item["id"] for item in test],
        "dev_type_counts": dict(Counter(item["type"] for item in dev)),
        "heldout_type_counts": dict(Counter(item["type"] for item in test)),
    }
    DEV_QUESTIONS.write_text(json.dumps(dev, indent=2) + "\n")
    TEST_QUESTIONS.write_text(json.dumps(test, indent=2) + "\n")
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def question_cost(model: str, usage: Any) -> float:
    rates = PRICING[model]
    cached = usage.cached_input_tokens
    writes = usage.cache_write_tokens
    ordinary = max(0, usage.input_tokens - cached - writes)
    return (
        ordinary * rates["input"]
        + cached * rates["cached"]
        + writes * rates["cache_write"]
        + usage.output_tokens * rates["output"]
    ) / 1_000_000


def evaluate_questions(
    questions: list[dict[str, str]],
    config: dict[str, Any],
    checkpoint: Path | None = None,
) -> list[dict[str, Any]]:
    agent = WaterDataAgent(
        DATABASE,
        model=config["model"],
        use_schema_guide=config["use_schema_guide"],
        use_examples=config["use_examples"],
        inspect_values_first=config["inspect_values_first"],
        service_tier=config["service_tier"],
    )
    rows: list[dict[str, Any]] = []
    if checkpoint and checkpoint.is_file():
        with checkpoint.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    completed = {row["id"] for row in rows}
    remaining = [item for item in questions if item["id"] not in completed]
    for index, item in enumerate(remaining, start=len(rows) + 1):
        expected = run_select(DATABASE, item["expected_sql"], row_limit=1000)
        answer = agent.ask(item["question"])
        correct = results_match(expected, answer.result)
        rows.append(
            {
                "config": config["name"],
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "expected_sql": item["expected_sql"],
                "expected_result": json.dumps(expected.rows, default=str),
                "agent_answer": answer.answer,
                "agent_sql": answer.sql or "",
                "agent_result": json.dumps(answer.result.rows if answer.result else None, default=str),
                "correct": str(correct).lower(),
                "failure_type": "" if correct else classify_failure(item["expected_sql"], answer.sql, answer.error),
                "error": answer.error or "",
                "attempts": answer.attempts,
                "model": config["model"],
                "service_tier": config["service_tier"],
                "input_tokens": answer.usage.input_tokens,
                "cached_input_tokens": answer.usage.cached_input_tokens,
                "cache_write_tokens": answer.usage.cache_write_tokens,
                "output_tokens": answer.usage.output_tokens,
                "api_requests": answer.usage.requests,
                "cost_usd": f"{question_cost(config['model'], answer.usage):.10f}",
            }
        )
        if checkpoint:
            _write_csv(checkpoint, rows)
        print(f"{config['name']}: {index}/{len(questions)}", flush=True)
    return rows


def summarize(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(row["correct"] == "true" for row in rows)
    summary: dict[str, Any] = {
        "config": config["name"],
        "model": config["model"],
        "schema_guide": config["use_schema_guide"],
        "examples": config["use_examples"],
        "distinct_check": config["inspect_values_first"],
        "service_tier": config["service_tier"],
        "correct": correct,
        "questions": len(rows),
        "accuracy": f"{correct / len(rows):.6f}",
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "cached_input_tokens": sum(int(row["cached_input_tokens"]) for row in rows),
        "cache_write_tokens": sum(int(row["cache_write_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_cost_usd": f"{sum(float(row['cost_usd']) for row in rows):.10f}",
        "cost_per_question_usd": f"{sum(float(row['cost_usd']) for row in rows) / len(rows):.10f}",
    }
    for question_type in ("simple_lookup", "aggregation", "year_comparison", "multi_table_join"):
        subset = [row for row in rows if row["type"] == question_type]
        hits = sum(row["correct"] == "true" for row in subset)
        summary[f"{question_type}_correct"] = hits
        summary[f"{question_type}_questions"] = len(subset)
        summary[f"{question_type}_accuracy"] = f"{hits / len(subset):.6f}"
    return summary


def run_dev(start_config: str | None = None) -> list[dict[str, Any]]:
    if not DEV_QUESTIONS.is_file():
        raise RuntimeError("Run --prepare-split first.")
    questions = json.loads(DEV_QUESTIONS.read_text())
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    recovered_path = RESULTS / "agent_dev_recovered_summaries.json"
    if recovered_path.is_file():
        summaries.extend(json.loads(recovered_path.read_text()))
    start_index = 0
    if start_config:
        start_index = next(i for i, item in enumerate(CONFIGS) if item["name"] == start_config)
    completed_configs = {row["config"] for row in summaries}
    for config in CONFIGS[start_index:]:
        if config["name"] in completed_configs:
            continue
        checkpoint = RESULTS / f"agent_dev_{config['name']}.csv"
        rows = evaluate_questions(questions, config, checkpoint=checkpoint)
        all_rows.extend(rows)
        summary = summarize(config, rows)
        summaries.append(summary)
        print(
            f"{config['name']} dev accuracy: {summary['correct']}/{summary['questions']} "
            f"({float(summary['accuracy']):.1%}); cost/question ${float(summary['cost_per_question_usd']):.6f}",
            flush=True,
        )
    for config in CONFIGS:
        checkpoint = RESULTS / f"agent_dev_{config['name']}.csv"
        if checkpoint.is_file():
            with checkpoint.open(newline="") as handle:
                all_rows.extend(list(csv.DictReader(handle)))
    if all_rows:
        deduplicated = {(row["config"], row["id"]): row for row in all_rows}
        _write_csv(DEV_DETAIL, list(deduplicated.values()))
    _write_csv(DEV_SUMMARY, summaries)
    return summaries


def _best_config() -> tuple[dict[str, Any], dict[str, str]]:
    with DEV_SUMMARY.open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    best = max(
        summaries,
        key=lambda row: (float(row["accuracy"]), -float(row["cost_per_question_usd"])),
    )
    config = next(item for item in CONFIGS if item["name"] == best["config"])
    return config, best


def _metrics(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    metrics = {"overall": (sum(row["correct"] == "true" for row in rows), len(rows))}
    for question_type in ("simple_lookup", "aggregation", "year_comparison", "multi_table_join"):
        subset = [row for row in rows if row["type"] == question_type]
        metrics[question_type] = (sum(row["correct"] == "true" for row in subset), len(subset))
    return metrics


def update_readme(
    baseline_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    dev_summaries: list[dict[str, str]],
    final_config: dict[str, Any],
) -> None:
    baseline = _metrics(baseline_rows)
    final = _metrics(final_rows)
    baseline_dev = next(row for row in dev_summaries if row["config"] == "baseline")
    dev_lines = "\n".join(
        f"- `{row['config']}` (`{row['model']}`, `{row['service_tier']}`): "
        f"**{int(row['correct'])}/{int(row['questions'])} "
        f"({float(row['accuracy']):.1%})**, **${float(row['cost_per_question_usd']):.6f}/question**"
        for row in dev_summaries
    )
    type_lines = "\n".join(
        f"- `{name}`: baseline **{baseline[name][0]}/{baseline[name][1]} "
        f"({baseline[name][0] / baseline[name][1]:.1%})**; final **{final[name][0]}/{final[name][1]} "
        f"({final[name][0] / final[name][1]:.1%})**"
        for name in ("simple_lookup", "aggregation", "year_comparison", "multi_table_join")
    )
    final_cost = sum(float(row["cost_usd"]) for row in final_rows) / len(final_rows)
    failures = Counter(
        row["failure_type"] for row in final_rows if row["correct"] != "true"
    )
    failure_text = ", ".join(
        f"{name.replace('_', ' ')}: {count}" for name, count in sorted(failures.items())
    ) or "none"
    generated = f"""<!-- AGENT_EVAL_START -->
## Chat-with-the-data agent evaluation

The benchmark was split with seed `{SPLIT_SEED}` into 20 development questions and 20 held-out
questions, with five questions of each type in each split. Only development results were used to
choose the final configuration. The four examples in the prompt are separate from all 40 benchmark
questions.

Development configurations actually run before the remaining experiments were skipped to conserve
requests:

{dev_lines}

The selected final configuration is `{final_config['name']}` using `{final_config['model']}`.
On the held-out questions, baseline accuracy from the pre-split archived run was
**{baseline['overall'][0]}/{baseline['overall'][1]} ({baseline['overall'][0] / baseline['overall'][1]:.1%})**;
the final agent's one test run scored **{final['overall'][0]}/{final['overall'][1]}
({final['overall'][0] / final['overall'][1]:.1%})**.

{type_lines}

Measured baseline cost on the development set was
**${float(baseline_dev['cost_per_question_usd']):.6f}/question**. Measured final-agent cost on the
held-out set was **${final_cost:.6f}/question**. Costs use recorded API tokens and the standard
short-context rates published at https://developers.openai.com/api/docs/pricing on 2026-10-09.
The final run's failure log contains: **{failure_text}**.

Development summaries are in `results/agent_dev_summary.csv`; the one-time final test is in
`results/agent_final_test.csv`; and baseline/final held-out outcomes are in
`results/agent_heldout_comparison.csv`.
<!-- AGENT_EVAL_END -->"""
    text = README.read_text()
    start = "<!-- AGENT_EVAL_START -->"
    end = "<!-- AGENT_EVAL_END -->"
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    README.write_text(before + generated + after)


def run_final_test() -> None:
    if FINAL_TEST.exists():
        raise RuntimeError(
            "Final held-out results already exist; refusing to run the held-out set more than once."
        )
    if not TEST_QUESTIONS.is_file():
        raise RuntimeError("Prepare the split first.")
    recovered_path = RESULTS / "agent_dev_recovered_summaries.standard.json"
    if not recovered_path.is_file():
        raise RuntimeError("The completed development-run summaries are missing.")
    dev_summaries = json.loads(recovered_path.read_text())
    for row in dev_summaries:
        row["service_tier"] = "default"
    _write_csv(DEV_SUMMARY, dev_summaries)
    config = dict(next(item for item in CONFIGS if item["name"] == "schema_guide"))
    config["min_request_interval_seconds"] = 2.0
    config["max_rate_limit_retries"] = 8
    questions = json.loads(TEST_QUESTIONS.read_text())
    final_rows = evaluate_questions(questions, config, checkpoint=FINAL_TEST_PARTIAL)
    _write_csv(FINAL_TEST, final_rows)
    FINAL_TEST_PARTIAL.unlink(missing_ok=True)
    FINAL_CONFIG.write_text(json.dumps(config, indent=2) + "\n")

    with BASELINE_FULL.open(newline="") as handle:
        baseline_all = list(csv.DictReader(handle))
    test_ids = {item["id"] for item in questions}
    baseline_rows = [row for row in baseline_all if row["id"] in test_ids]
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    comparison = []
    for row in final_rows:
        old = baseline_by_id[row["id"]]
        comparison.append(
            {
                "id": row["id"],
                "type": row["type"],
                "baseline_correct": old["correct"],
                "final_correct": row["correct"],
                "baseline_model": old["model"],
                "final_model": row["model"],
                "final_cost_usd": row["cost_usd"],
            }
        )
    _write_csv(HELDOUT_COMPARISON, comparison)
    update_readme(baseline_rows, final_rows, dev_summaries, config)
    baseline_metrics = _metrics(baseline_rows)["overall"]
    final_metrics = _metrics(final_rows)["overall"]
    print(
        f"Held-out baseline: {baseline_metrics[0]}/{baseline_metrics[1]} "
        f"({baseline_metrics[0] / baseline_metrics[1]:.1%})"
    )
    print(
        f"Held-out final: {final_metrics[0]}/{final_metrics[1]} "
        f"({final_metrics[0] / final_metrics[1]:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-split", action="store_true")
    action.add_argument("--run-dev", action="store_true")
    action.add_argument("--run-final-test", action="store_true")
    parser.add_argument("--start-config", choices=[item["name"] for item in CONFIGS])
    args = parser.parse_args()
    if args.prepare_split:
        manifest = prepare_split()
        print(json.dumps(manifest, indent=2))
    elif args.run_dev:
        try:
            run_dev(args.start_config)
        except Exception as exc:
            if exc.__class__.__name__ == "RateLimitError":
                print(f"RATE_LIMIT: {exc}", flush=True)
                raise SystemExit(75) from None
            raise
    else:
        try:
            run_final_test()
        except Exception as exc:
            if exc.__class__.__name__ == "RateLimitError":
                print(f"RATE_LIMIT: {exc}", flush=True)
                raise SystemExit(75) from None
            raise


if __name__ == "__main__":
    main()
