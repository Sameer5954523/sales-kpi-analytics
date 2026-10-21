"""Load the cleaned data into SQLite and run sql/kpi_queries.sql, printing each result."""

import sqlite3
from pathlib import Path

import pandas as pd

from analysis import load_and_clean

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    df, _ = load_and_clean()
    con = sqlite3.connect(":memory:")
    cols = ["lead_id", "created_at", "source", "campaign", "agent_id", "site", "minutes_to_first_contact",
            "contacted", "qualified", "converted", "revenue_gbp", "acquisition_cost_gbp"]
    out = df[cols].copy()
    out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ["contacted", "qualified", "converted"]:
        out[c] = out[c].astype(int)
    out.to_sql("leads", con, index=False)
    pd.read_csv(ROOT / "data" / "agents.csv").to_sql("agents", con, index=False)
    sql = (ROOT / "sql" / "kpi_queries.sql").read_text()
    for i, stmt in enumerate([s for s in sql.split(";") if "SELECT" in s.upper()], start=1):
        print(f"\n-- Query {i}")
        print(pd.read_sql_query(stmt, con).to_string(index=False))
