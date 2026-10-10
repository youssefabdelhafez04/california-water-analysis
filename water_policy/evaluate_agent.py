"""Run the 40-question execution-accuracy evaluation for the SQL agent."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from water_policy.chat_agent import QueryResult, WaterDataAgent, run_select


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "results" / "water_analysis.sqlite"
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions.json"
DEFAULT_OUTPUT = ROOT / "results" / "agent_eval.csv"
DEFAULT_GROUND_TRUTH = ROOT / "results" / "agent_eval_test_set.csv"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict[str, str]]:
    questions = json.loads(path.read_text())
    required_types = {"simple_lookup", "aggregation", "year_comparison", "multi_table_join"}
    counts = Counter(item["type"] for item in questions)
    if len(questions) != 40 or set(counts) != required_types or any(counts[t] != 10 for t in required_types):
        raise ValueError(f"Evaluation set must contain 10 questions of each type; found {dict(counts)}")
    if len({item["id"] for item in questions}) != 40:
        raise ValueError("Evaluation question IDs must be unique.")
    return questions


def _equal_scalar(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)
    return str(left) == str(right)


def results_match(expected: QueryResult, actual: QueryResult | None) -> bool:
    """Execution accuracy: same result shape and values, with small float tolerance."""
    if actual is None or len(expected.rows) != len(actual.rows):
        return False
    if len(expected.columns) != len(actual.columns):
        return False
    # Every benchmark query specifies a meaningful ordering when it returns multiple rows.
    return all(
        len(expected_row) == len(actual_row)
        and all(_equal_scalar(e, a) for e, a in zip(expected_row, actual_row))
        for expected_row, actual_row in zip(expected.rows, actual.rows)
    )


def _tables(sql: str) -> set[str]:
    return {match.lower() for match in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", sql, re.I)}


def classify_failure(expected_sql: str, agent_sql: str | None, error: str | None) -> str:
    if error or not agent_sql:
        return "query_error"
    if _tables(expected_sql) != _tables(agent_sql):
        return "wrong_table"
    expected_aggregates = {
        value.upper() for value in re.findall(r"\b(?:AVG|SUM|COUNT|MIN|MAX)\s*\(", expected_sql, re.I)
    }
    actual_aggregates = {
        value.upper() for value in re.findall(r"\b(?:AVG|SUM|COUNT|MIN|MAX)\s*\(", agent_sql, re.I)
    }
    if expected_aggregates != actual_aggregates:
        return "wrong_aggregation"
    expected_literals = set(re.findall(r"(?:\b(?:19|20)\d{2}\b|'[^']+')", expected_sql))
    actual_literals = set(re.findall(r"(?:\b(?:19|20)\d{2}\b|'[^']+')", agent_sql))
    if not expected_literals.issubset(actual_literals):
        return "wrong_filter"
    return "other_result_mismatch"


def materialize_ground_truth(
    database: Path = DEFAULT_DATABASE,
    questions_path: Path = DEFAULT_QUESTIONS,
    output: Path = DEFAULT_GROUND_TRUTH,
) -> Path:
    """Execute every hand-authored reference SQL query and save its checked result."""
    rows = []
    for item in load_questions(questions_path):
        expected = run_select(database, item["expected_sql"], row_limit=1000)
        if expected.truncated:
            raise ValueError(f"Ground truth unexpectedly exceeded 1,000 rows: {item['id']}")
        rows.append(
            {
                **item,
                "expected_columns": json.dumps(expected.columns),
                "expected_result": json.dumps(expected.rows, default=str),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def update_readme(readme: Path, rows: list[dict[str, Any]], model: str) -> None:
    total = len(rows)
    correct = sum(row["correct"] == "true" for row in rows)
    by_type = {}
    for question_type in ("simple_lookup", "aggregation", "year_comparison", "multi_table_join"):
        subset = [row for row in rows if row["type"] == question_type]
        hits = sum(row["correct"] == "true" for row in subset)
        by_type[question_type] = (hits, len(subset))
    failures = Counter(row["failure_type"] for row in rows if row["failure_type"])
    failure_text = ", ".join(f"{name}: **{count}**" for name, count in sorted(failures.items())) or "none"
    type_lines = "\n".join(
        f"- `{name}`: **{hits}/{count} ({hits / count:.1%})**"
        for name, (hits, count) in by_type.items()
    )
    generated = f"""<!-- AGENT_EVAL_START -->
## Chat-with-the-data agent evaluation

The measured execution accuracy for `{model}` is **{correct}/{total} ({correct / total:.1%})**.
The benchmark contains 40 hand-authored questions whose correct results are executed directly
from reference SQL in `evaluation/questions.json`.

{type_lines}

Failure classifications: {failure_text}. Full per-question outputs, generated SQL, returned
results, errors, and classifications are in `results/agent_eval.csv`.
<!-- AGENT_EVAL_END -->"""
    text = readme.read_text()
    start = "<!-- AGENT_EVAL_START -->"
    end = "<!-- AGENT_EVAL_END -->"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _, after = remainder.split(end, 1)
        text = before + generated + after
    else:
        text = text.rstrip() + "\n\n" + generated + "\n"
    readme.write_text(text)


def evaluate(
    database: Path = DEFAULT_DATABASE,
    questions_path: Path = DEFAULT_QUESTIONS,
    output: Path = DEFAULT_OUTPUT,
    readme: Path = ROOT / "README.md",
) -> list[dict[str, Any]]:
    questions = load_questions(questions_path)
    agent = WaterDataAgent(database)
    rows: list[dict[str, Any]] = []
    for item in questions:
        expected = run_select(database, item["expected_sql"], row_limit=1000)
        answer = agent.ask(item["question"])
        correct = results_match(expected, answer.result)
        failure_type = "" if correct else classify_failure(item["expected_sql"], answer.sql, answer.error)
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "expected_sql": item["expected_sql"],
                "expected_result": json.dumps(expected.rows, default=str),
                "agent_answer": answer.answer,
                "agent_sql": answer.sql or "",
                "agent_result": json.dumps(answer.result.rows if answer.result else None, default=str),
                "correct": str(correct).lower(),
                "failure_type": failure_type,
                "error": answer.error or "",
                "attempts": answer.attempts,
                "model": agent.model,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    update_readme(readme, rows, agent.model)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-only", action="store_true")
    args = parser.parse_args()
    ground_truth_path = materialize_ground_truth()
    print(f"Wrote checked ground truth: {ground_truth_path}")
    if args.ground_truth_only:
        return
    rows = evaluate()
    correct = sum(row["correct"] == "true" for row in rows)
    print(f"Execution accuracy: {correct}/{len(rows)} ({correct / len(rows):.1%})")


if __name__ == "__main__":
    main()
