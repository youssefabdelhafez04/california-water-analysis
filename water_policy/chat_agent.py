"""Read-only, tool-calling SQL agent for the California water database."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_ROW_LIMIT = 100
DEFAULT_REQUEST_INTERVAL_SECONDS = 2.0
DEFAULT_RATE_LIMIT_RETRIES = 8


class SQLSafetyError(ValueError):
    """Raised before execution when SQL violates the read-only policy."""


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "rows": self.rows, "truncated": self.truncated}


@dataclass
class AgentAnswer:
    question: str
    answer: str
    sql: str | None
    result: QueryResult | None
    error: str | None = None
    attempts: int = 0
    usage: "TokenUsage" = field(default_factory=lambda: TokenUsage())


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def add_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.requests += 1
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        self.cached_input_tokens += int(getattr(details, "cached_tokens", 0) or 0)
        self.cache_write_tokens += int(getattr(details, "cache_write_tokens", 0) or 0)


_BLOCKED_KEYWORDS = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|TRUNCATE|VACUUM|"
    r"ATTACH|DETACH|PRAGMA|REINDEX|ANALYZE|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b",
    re.IGNORECASE,
)


def _sql_without_literals_or_comments(sql: str) -> str:
    """Remove quoted content/comments before conservative keyword checks."""
    output: list[str] = []
    i = 0
    while i < len(sql):
        if sql.startswith("--", i):
            i = sql.find("\n", i)
            if i == -1:
                break
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end == -1:
                raise SQLSafetyError("Unterminated SQL comment.")
            i = end + 2
        elif sql[i] in {"'", '"'}:
            quote = sql[i]
            output.append(" ")
            i += 1
            while i < len(sql):
                if sql[i] == quote:
                    if i + 1 < len(sql) and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                raise SQLSafetyError("Unterminated SQL string.")
        else:
            output.append(sql[i])
            i += 1
    return "".join(output)


def validate_select_sql(sql: str) -> str:
    """Accept one SELECT (optionally beginning with WITH) and reject writes/PRAGMAs."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLSafetyError("The query is empty.")
    cleaned = _sql_without_literals_or_comments(sql).strip()
    statements = [part for part in cleaned.split(";") if part.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("Only one SQL statement is allowed.")
    first = re.match(r"^([A-Za-z]+)", statements[0])
    if first is None or first.group(1).upper() not in {"SELECT", "WITH"}:
        raise SQLSafetyError("Only SELECT queries are allowed.")
    if _BLOCKED_KEYWORDS.search(cleaned):
        raise SQLSafetyError("Only read-only SELECT queries are allowed.")
    return sql.strip().rstrip(";").strip()


def _authorizer(action: int, _arg1: str | None, arg2: str | None, _db: str | None, _source: str | None) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() == "load_extension":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def open_readonly_database(database_path: str | Path) -> sqlite3.Connection:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(_authorizer)
    return connection


def run_select(database_path: str | Path, sql: str, row_limit: int = DEFAULT_ROW_LIMIT) -> QueryResult:
    """Run validated SQL through a read-only connection and enforce a hard row cap."""
    if row_limit < 1:
        raise ValueError("row_limit must be positive")
    safe_sql = validate_select_sql(sql)
    limited_sql = f"SELECT * FROM ({safe_sql}) AS agent_query LIMIT {int(row_limit) + 1}"
    with closing(open_readonly_database(database_path)) as connection:
        cursor = connection.execute(limited_sql)
        columns = [item[0] for item in cursor.description or []]
        raw_rows = cursor.fetchall()
    truncated = len(raw_rows) > row_limit
    rows = [list(row) for row in raw_rows[:row_limit]]
    return QueryResult(columns=columns, rows=rows, truncated=truncated)


def database_schema(database_path: str | Path) -> str:
    """Return table/column metadata, excluding SQLite internals."""
    sql = """
        SELECT m.name AS table_name, p.name AS column_name, p.type AS data_type
        FROM sqlite_master AS m
        JOIN pragma_table_info(m.name) AS p
        WHERE m.type = 'table' AND m.name NOT LIKE 'sqlite_%'
        ORDER BY m.name, p.cid
    """
    # Schema introspection is application-owned and runs before the query authorizer.
    path = Path(database_path).resolve()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        rows = connection.execute(sql).fetchall()
    grouped: dict[str, list[str]] = {}
    for table, column, data_type in rows:
        grouped.setdefault(table, []).append(f"{column} {data_type or 'UNKNOWN'}")
    return "\n".join(f"{table}({', '.join(columns)})" for table, columns in grouped.items())


def database_columns(database_path: str | Path) -> dict[str, set[str]]:
    path = Path(database_path).resolve()
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {
            table: {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
            for (table,) in tables
        }


def inspect_distinct_values(
    database_path: str | Path, table: str, column: str, row_limit: int = DEFAULT_ROW_LIMIT
) -> QueryResult:
    columns = database_columns(database_path)
    if table not in columns or column not in columns[table]:
        raise SQLSafetyError("The requested table or column does not exist.")
    sql = (
        f'SELECT DISTINCT "{column}" FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL ORDER BY 1'
    )
    return run_select(database_path, sql, row_limit=row_limit)


class WaterDataAgent:
    """OpenAI Responses API agent with one SQL tool and at most one error retry."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        model: str | None = None,
        row_limit: int = DEFAULT_ROW_LIMIT,
        client: Any | None = None,
        use_schema_guide: bool | None = None,
        use_examples: bool | None = None,
        inspect_values_first: bool | None = None,
        service_tier: str | None = None,
        min_request_interval_seconds: float | None = None,
        max_rate_limit_retries: int | None = None,
        sleep_fn: Any = time.sleep,
        clock_fn: Any = time.monotonic,
    ) -> None:
        self.database_path = Path(database_path)
        config_path = self.database_path.parent / "agent_final_config.json"
        saved_config: dict[str, Any] = {}
        if config_path.is_file():
            saved_config = json.loads(config_path.read_text())
        self.model = (
            model
            or os.environ.get("OPENAI_MODEL")
            or saved_config.get("model")
            or DEFAULT_MODEL
        )
        self.row_limit = row_limit
        self.use_schema_guide = (
            saved_config.get("use_schema_guide", False)
            if use_schema_guide is None
            else use_schema_guide
        )
        self.use_examples = (
            saved_config.get("use_examples", False) if use_examples is None else use_examples
        )
        self.inspect_values_first = (
            saved_config.get("inspect_values_first", False)
            if inspect_values_first is None
            else inspect_values_first
        )
        self.service_tier = service_tier or saved_config.get("service_tier") or "auto"
        configured_interval = saved_config.get(
            "min_request_interval_seconds", DEFAULT_REQUEST_INTERVAL_SECONDS
        )
        self.min_request_interval_seconds = float(
            os.environ.get(
                "OPENAI_MIN_REQUEST_INTERVAL_SECONDS",
                configured_interval if min_request_interval_seconds is None else min_request_interval_seconds,
            )
        )
        if self.min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds cannot be negative")
        self.max_rate_limit_retries = int(
            saved_config.get("max_rate_limit_retries", DEFAULT_RATE_LIMIT_RETRIES)
            if max_rate_limit_retries is None
            else max_rate_limit_retries
        )
        if self.max_rate_limit_retries < 0:
            raise ValueError("max_rate_limit_retries cannot be negative")
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._last_request_at: float | None = None
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it before using Chat with the data."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install the 'openai' package to use the data agent.") from exc
            # Disable the SDK's hidden retries so every retry is paced and accounted for here.
            client = OpenAI(max_retries=0)
        self.client = client
        self.schema = database_schema(self.database_path)

    def _sleep_for(self, seconds: float) -> None:
        """Sleep in bounded chunks so long rate-limit waits remain interruptible."""
        remaining = max(0.0, seconds)
        while remaining > 0:
            chunk = min(remaining, 55.0)
            self._sleep(chunk)
            remaining -= chunk

    @staticmethod
    def _retry_delay(error: Exception) -> float:
        """Prefer Retry-After/reset metadata, then parse the API's human-readable delay."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or getattr(error, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        message = str(error)
        match = re.search(
            r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(\d+(?:\.\d+)?)s",
            message,
            re.IGNORECASE,
        )
        if match:
            return (
                int(match.group(1) or 0) * 3600
                + int(match.group(2) or 0) * 60
                + float(match.group(3))
            )
        match = re.search(r"reset[^\d]*(\d+(?:\.\d+)?)s", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 60.0

    def _create_response(self, **kwargs: Any) -> Any:
        """Pace API requests and retry rate-limit failures using the server delay."""
        retries = 0
        while True:
            if self._last_request_at is not None:
                elapsed = self._clock() - self._last_request_at
                self._sleep_for(self.min_request_interval_seconds - elapsed)
            self._last_request_at = self._clock()
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                if exc.__class__.__name__ != "RateLimitError" or retries >= self.max_rate_limit_retries:
                    raise
                retries += 1
                self._sleep_for(max(self.min_request_interval_seconds, self._retry_delay(exc)))

    @property
    def instructions(self) -> str:
        instructions = f"""You answer questions only from the California water SQLite database.
Use run_sql for every data question. Write SQLite SELECT queries only. Do not guess.
If the tool returns an error, correct the query and retry once. If the data cannot answer
the question, say so clearly. After a successful tool result, give a short plain-English
answer based strictly on the returned values. Do not round values inside SQL unless asked.
The application separately displays the SQL and result table.

Schema:
{self.schema}
"""
        if self.use_schema_guide:
            from water_policy.agent_prompt import SCHEMA_GUIDE

            instructions += "\n\n" + SCHEMA_GUIDE
        if self.use_examples:
            from water_policy.agent_prompt import FEW_SHOT_EXAMPLES

            instructions += "\n\n" + FEW_SHOT_EXAMPLES
        if self.inspect_values_first:
            instructions += """

Before writing the answer query, call inspect_distinct_values once on the categorical or
year column most important to the question. Use the returned exact stored spelling in run_sql.
The inspection is only preparation; always follow it with run_sql for the actual answer.
"""
        return instructions

    @property
    def tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "type": "function",
                "name": "run_sql",
                "description": "Run one read-only SQLite SELECT query against the water database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "A single SQLite SELECT query."}
                    },
                    "required": ["sql"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
        if self.inspect_values_first:
            tools.append(
                {
                    "type": "function",
                    "name": "inspect_distinct_values",
                    "description": "Inspect exact stored values in one database column before filtering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Exact table name."},
                            "column": {"type": "string", "description": "Exact column name."},
                        },
                        "required": ["table", "column"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            )
        return tools

    def ask(self, question: str) -> AgentAnswer:
        if not question.strip():
            return AgentAnswer(question, "Please enter a question about the data.", None, None)

        conversation: list[Any] = [{"role": "user", "content": question.strip()}]
        last_sql: str | None = None
        last_result: QueryResult | None = None
        last_error: str | None = None
        attempts = 0
        usage = TokenUsage()
        inspection_done = not self.inspect_values_first

        # Optional value inspection, answer query, one query retry, and final response.
        for _ in range(5):
            if not inspection_done:
                tool_choice: Any = {"type": "function", "name": "inspect_distinct_values"}
            elif last_result is None:
                tool_choice = {"type": "function", "name": "run_sql"} if self.inspect_values_first else "required"
            else:
                tool_choice = "none"
            response = self._create_response(
                model=self.model,
                instructions=self.instructions,
                tools=self.tools,
                tool_choice=tool_choice,
                input=conversation,
                store=False,
                service_tier=self.service_tier,
            )
            usage.add_response(response)
            conversation.extend(response.output)
            calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not calls:
                answer = (response.output_text or "").strip()
                if last_result is not None:
                    return AgentAnswer(
                        question, answer, last_sql, last_result, attempts=attempts, usage=usage
                    )
                message = answer or "I couldn't answer that question from the available data."
                return AgentAnswer(question, message, last_sql, None, last_error, attempts, usage)

            call = calls[0]
            arguments = json.loads(call.arguments)
            if getattr(call, "name", "run_sql") == "inspect_distinct_values":
                try:
                    inspected = inspect_distinct_values(
                        self.database_path, arguments["table"], arguments["column"], self.row_limit
                    )
                    inspection_done = True
                    payload = {"ok": True, **inspected.as_dict()}
                except (KeyError, TypeError, sqlite3.Error, SQLSafetyError) as exc:
                    inspection_done = True
                    payload = {"ok": False, "error": str(exc)}
            else:
                if attempts >= 2:
                    break
                attempts += 1
                try:
                    last_sql = arguments["sql"]
                    last_result = run_select(self.database_path, last_sql, self.row_limit)
                    last_error = None
                    payload = {"ok": True, **last_result.as_dict()}
                except (KeyError, TypeError, sqlite3.Error, SQLSafetyError) as exc:
                    last_result = None
                    last_error = str(exc)
                    payload = {
                        "ok": False,
                        "error": last_error,
                        "instruction": "Fix the SQL and retry once, or explain that the question cannot be answered.",
                    }
            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(payload, default=str),
                }
            )
            if last_error and attempts >= 2:
                return AgentAnswer(
                    question,
                    f"I couldn't run a valid query after one retry: {last_error}",
                    last_sql,
                    None,
                    last_error,
                    attempts,
                    usage,
                )

        if last_error:
            message = f"I couldn't run a valid query after one retry: {last_error}"
        else:
            message = "I couldn't produce a database-backed answer for that question."
        return AgentAnswer(question, message, last_sql, last_result, last_error, attempts, usage)
