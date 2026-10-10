import sqlite3
from types import SimpleNamespace

import pytest

from water_policy.chat_agent import (
    SQLSafetyError,
    WaterDataAgent,
    open_readonly_database,
    run_select,
    validate_select_sql,
)
from water_policy.evaluate_agent import load_questions, materialize_ground_truth


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "test.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE observations (name TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO observations VALUES (?, ?)", [("a", 1.0), ("b", 2.0), ("c", 3.0)]
        )
    return path


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM observations",
        "UPDATE observations SET value = 0",
        "INSERT INTO observations VALUES ('x', 4)",
        "DROP TABLE observations",
        "PRAGMA table_info(observations)",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "SELECT * FROM observations; DELETE FROM observations",
        "WITH changed AS (DELETE FROM observations RETURNING *) SELECT * FROM changed",
    ],
)
def test_rejects_non_select_queries(sql):
    with pytest.raises(SQLSafetyError, match="SELECT|statement|read-only"):
        validate_select_sql(sql)


def test_select_runs_with_hard_row_limit(database):
    result = run_select(database, "SELECT name, value FROM observations ORDER BY value", row_limit=2)
    assert result.columns == ["name", "value"]
    assert result.rows == [["a", 1.0], ["b", 2.0]]
    assert result.truncated is True


def test_readonly_connection_cannot_write_even_without_validator(database):
    with open_readonly_database(database) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM observations")


def test_keywords_inside_string_literals_are_allowed(database):
    result = run_select(database, "SELECT 'DROP TABLE observations' AS text")
    assert result.rows == [["DROP TABLE observations"]]


def test_agent_gets_one_retry_after_query_error(database):
    responses = [
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    arguments='{"sql":"SELECT missing FROM observations"}',
                    call_id="first",
                )
            ],
            output_text="",
        ),
        SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    arguments='{"sql":"SELECT MAX(value) FROM observations"}',
                    call_id="second",
                )
            ],
            output_text="",
        ),
        SimpleNamespace(output=[], output_text="The maximum is 3."),
    ]

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return responses[len(self.calls) - 1]

    fake = SimpleNamespace(responses=FakeResponses())
    answer = WaterDataAgent(
        database, client=fake, min_request_interval_seconds=0
    ).ask("What is the maximum value?")
    assert answer.attempts == 2
    assert answer.result.rows == [[3.0]]
    assert answer.sql == "SELECT MAX(value) FROM observations"
    assert answer.answer == "The maximum is 3."
    assert [call["tool_choice"] for call in fake.responses.calls] == ["required", "required", "none"]


def test_agent_waits_and_retries_rate_limit(database):
    class RateLimitError(Exception):
        pass

    successful = SimpleNamespace(output=[], output_text="No database answer.")

    class FakeResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("Rate limit reached; try again in 1m 2.5s")
            return successful

    sleeps = []
    fake = SimpleNamespace(responses=FakeResponses())
    agent = WaterDataAgent(
        database,
        client=fake,
        min_request_interval_seconds=2,
        max_rate_limit_retries=1,
        sleep_fn=sleeps.append,
        clock_fn=lambda: 0.0,
    )
    answer = agent.ask("Can this be answered?")
    assert answer.answer == "No database answer."
    assert fake.responses.calls == 2
    assert sleeps == [55.0, 7.5, 2.0]
    assert agent._retry_delay(RateLimitError("try again in 1h 2m 3s")) == 3723


def test_eval_set_has_40_executable_reference_queries(tmp_path):
    questions = load_questions()
    assert len(questions) == 40
    # The repository database is used here because the benchmark targets its real schema.
    output = tmp_path / "ground_truth.csv"
    materialize_ground_truth(output=output)
    assert output.read_text().count("\n") == 41
