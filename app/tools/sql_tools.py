"""
SQL Skill.

Lets the agent (or the user, via natural language the LLM translates
into SQL) run read-only queries against the uploaded dataset. DuckDB
queries the CSV directly - no database server or ETL step needed -
which keeps this simple while still giving real SQL semantics
(GROUP BY, window functions, CTEs, etc.).

Safety model:
- Only a single SELECT (optionally preceded by WITH ... AS (...)) is
  allowed. Anything else is rejected before it ever reaches DuckDB.
- A keyword blocklist catches destructive/DDL statements even if they
  are smuggled inside a CTE or subquery.
- Multiple statements (via a semicolon) are rejected.
- Results are capped so a huge SELECT * can't blow up the response.
"""

import re

import duckdb
import pandas as pd


MAX_ROWS_RETURNED = 200

_ALLOWED_START = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)

_FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|"
    r"ATTACH|DETACH|COPY|PRAGMA|CALL|EXPORT|IMPORT|VACUUM|GRANT|REVOKE|"
    r"INSTALL|LOAD"
    r")\b",
    re.IGNORECASE,
)


def validate_sql_query(sql_query: str) -> dict:
    """Validate that a query is a single, read-only SELECT."""

    if not sql_query or not sql_query.strip():
        return {"valid": False, "error": "Empty SQL query."}

    query = sql_query.strip()

    # Strip one trailing semicolon (a single statement is still fine),
    # but reject anything with a semicolon in the middle - that would
    # mean multiple statements.
    if query.endswith(";"):
        query = query[:-1].rstrip()
    if ";" in query:
        return {
            "valid": False,
            "error": "Multiple SQL statements are not allowed.",
        }

    if not _ALLOWED_START.match(query):
        return {
            "valid": False,
            "error": "Only SELECT (or WITH ... SELECT) queries are allowed.",
        }

    if _FORBIDDEN_KEYWORDS.search(query):
        return {
            "valid": False,
            "error": (
                "Query contains a disallowed keyword. Only read-only "
                "SELECT queries are permitted - no INSERT, UPDATE, "
                "DELETE, DROP, ALTER, or similar statements."
            ),
        }

    return {"valid": True, "query": query}


def run_sql_query(datasets: dict, sql_query: str) -> dict:
    """
    Run a validated, read-only SQL query across the session's
    uploaded datasets.

    `datasets` maps table_name -> CSV file path; every dataset in the
    session is registered as its own DuckDB table under that name, so
    a single query can JOIN across them, e.g.:

        SELECT u.country, SUM(oi.sale_price) AS revenue
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN users_old u ON o.user_id = u.id
        GROUP BY u.country
        ORDER BY revenue DESC
        LIMIT 5
    """

    validation = validate_sql_query(sql_query)
    if not validation["valid"]:
        return {"error": validation["error"]}

    if not datasets:
        return {"error": "No datasets are available to query."}

    connection = None
    try:
        connection = duckdb.connect(database=":memory:")
        for table_name, file_path in datasets.items():
            try:
                df = pd.read_csv(file_path)
            except Exception as exc:
                return {
                    "error": (
                        f"Unable to read dataset '{table_name}': {exc}"
                    )
                }

            # pandas >= 3.0 defaults text columns to its new 'str'
            # extension dtype instead of the classic 'object' dtype.
            # duckdb 1.5.5 predates that dtype and can't register a
            # frame containing it ("Not implemented Error: Data type
            # 'str' not recognized"), which surfaces as a confusing
            # failure on any query that touches a text column (e.g.
            # a JOIN or GROUP BY on a string key). Downcast back to
            # plain 'object' first, which duckdb has always supported.
            for column in df.columns:
                if str(df[column].dtype) in ("str", "string"):
                    df[column] = df[column].astype(object)

            connection.register(table_name, df)

        result_df = connection.execute(validation["query"]).fetchdf()
    except Exception as exc:
        return {"error": f"SQL execution failed: {exc}"}
    finally:
        if connection is not None:
            connection.close()

    total_rows = int(len(result_df))
    truncated = total_rows > MAX_ROWS_RETURNED
    limited_df = result_df.head(MAX_ROWS_RETURNED)

    return {
        "status": "success",
        "query": validation["query"],
        "tables_available": list(datasets.keys()),
        "row_count": total_rows,
        "returned_rows": int(len(limited_df)),
        "truncated": truncated,
        "columns": limited_df.columns.tolist(),
        "rows": limited_df.to_dict(orient="records"),
    }