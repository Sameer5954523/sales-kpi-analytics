"""End-to-end analysis: clean the raw export, compute KPIs, build charts and a written summary."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, REPORTS = ROOT / "data", ROOT / "reports" / "figures", ROOT / "reports"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
ACCENT, MUTED = "#1F4E79", "#9DB4CC"


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(DATA / "leads_raw.csv", parse_dates=["created_at"])
    agents = pd.read_csv(DATA / "agents.csv")
    log = {"raw_rows": len(raw)}
    df = raw.drop_duplicates(subset="lead_id").copy()
    log["duplicates_removed"] = log["raw_rows"] - len(df)
    df["source"] = df["source"].str.strip().str.lower()
    bad = df["minutes_to_first_contact"] < 0
    log["invalid_contact_times"] = int(bad.sum())
    df.loc[bad, "minutes_to_first_contact"] = pd.NA
    df = df.merge(agents[["agent_id", "tenure_months"]], on="agent_id", how="left")
    df["month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = df["created_at"].dt.day_name()
    df["speed_bucket"] = pd.cut(df["minutes_to_first_contact"], [-0.1, 5, 30, 120, 10_000],
                                labels=["<=5 min", "6-30 min", "31-120 min", ">2 hours"])
    df["tenure_bucket"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 100], labels=["0-6 mo", "7-12 mo", "13-24 mo", "24+ mo"])
    log["clean_rows"] = len(df)
    return df, log


def kpis(df: pd.DataFrame) -> dict:
    conv = df["converted"].mean()
    rev, cost = df["revenue_gbp"].sum(), df["acquisition_cost_gbp"].sum()
    return {"leads": len(df), "contact_rate": df["contacted"].mean(), "qualification_rate": df["qualified"].mean(),
            "conversion_rate": conv, "revenue": rev, "acq_cost": cost, "roas": rev / cost,
            "median_minutes_to_contact": df["minutes_to_first_contact"].median()}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


def charts_and_tables(df: pd.DataFrame) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}

    funnel = pd.Series({"Leads": len(df), "Contacted": df["contacted"].sum(), "Qualified": df["qualified"].sum(), "Converted": df["converted"].sum()})
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(funnel.index[::-1], funnel.values[::-1], color=[ACCENT, MUTED, MUTED, MUTED])
    for i, v in enumerate(funnel.values[::-1]):
        ax.text(v, i, f" {v:,} ({v / len(df):.0%})", va="center")
    ax.set_title("Lead funnel")
    ax.set_xlim(0, len(df) * 1.25)
    save(fig, "01_funnel.png")

    src = df.groupby("source").agg(leads=("lead_id", "size"), conv_rate=("converted", "mean"),
                                   revenue=("revenue_gbp", "sum"), cost=("acquisition_cost_gbp", "sum"))
    src["cost_per_sale"] = src["cost"] / df[df["converted"]].groupby("source").size()
    src["roas"] = (src["revenue"] / src["cost"]).where(src["cost"] > 0)
    src = src.sort_values("conv_rate", ascending=False)
    out["source"] = src
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(src.index, src["conv_rate"] * 100, color=ACCENT)
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Conversion rate by lead source")
    save(fig, "02_source_conversion.png")

    speed = df.groupby("speed_bucket", observed=True)["converted"].agg(["mean", "size"])
    out["speed"] = speed
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(speed.index.astype(str), speed["mean"] * 100, color=[ACCENT, MUTED, MUTED, MUTED])
    ax.set_ylabel("Conversion rate (%)")
    ax.set_title("Speed-to-lead: conversion by time to first contact")
    save(fig, "03_speed_to_lead.png")

    monthly = df.groupby("month").agg(leads=("lead_id", "size"), conv=("converted", "mean"), revenue=("revenue_gbp", "sum"))
    out["monthly"] = monthly
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.bar(monthly.index, monthly["revenue"] / 1000, width=20, color=MUTED, label="Revenue (GBP k)")
    ax1.set_ylabel("Revenue (GBP k)")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["conv"] * 100, color=ACCENT, marker="o", label="Conversion %")
    ax2.set_ylabel("Conversion rate (%)")
    ax2.grid(False)
    ax1.set_title("Monthly revenue and conversion rate")
    save(fig, "04_monthly_trend.png")

    tenure = df.groupby("tenure_bucket", observed=True)["converted"].mean()
    site = df.groupby("site").agg(leads=("lead_id", "size"), conv_rate=("converted", "mean"),
                                  median_contact_min=("minutes_to_first_contact", "median")).sort_values("conv_rate", ascending=False)
    out["tenure"], out["site"] = tenure, site
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(tenure.index.astype(str), tenure * 100, color=ACCENT)
    axes[0].set_title("Conversion by agent tenure")
    axes[0].set_ylabel("Conversion rate (%)")
    axes[1].bar(site.index, site["conv_rate"] * 100, color=MUTED)
    axes[1].set_title("Conversion by site")
    save(fig, "05_tenure_and_site.png")

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = df.groupby("weekday")["converted"].mean().reindex(order)
    out["weekday"] = wd
    agent = df.groupby("agent_id").agg(leads=("lead_id", "size"), conv_rate=("converted", "mean"), revenue=("revenue_gbp", "sum"))
    out["top_agents"] = agent.sort_values("revenue", ascending=False).head(5)
    return out


def pct(x: float) -> str:
    return f"{x:.1%}"


def write_report(k: dict, t: dict, log: dict) -> None:
    src, speed = t["source"], t["speed"]
    best, worst = src.index[0], src.index[-1]
    fast, slow = speed["mean"].iloc[0], speed["mean"].iloc[-1]
    wd = t["weekday"]
    lines = [
        "# Findings summary",
        "",
        "_Generated by `src/analysis.py` from the synthetic dataset in `data/`._",
        "",
        "## Data cleaning",
        f"- Raw rows: {log['raw_rows']:,}. Removed {log['duplicates_removed']} duplicate lead records.",
        f"- Normalised inconsistent source labels (e.g. `FACEBOOK` to `facebook`).",
        f"- Set {log['invalid_contact_times']} impossible negative contact times to missing.",
        f"- Clean rows analysed: {log['clean_rows']:,}.",
        "",
        "## Headline KPIs",
        "| KPI | Value |", "|---|---|",
        f"| Leads | {k['leads']:,} |",
        f"| Contact rate | {pct(k['contact_rate'])} |",
        f"| Qualification rate | {pct(k['qualification_rate'])} |",
        f"| Conversion rate | {pct(k['conversion_rate'])} |",
        f"| Revenue | GBP {k['revenue']:,.0f} |",
        f"| Paid acquisition cost | GBP {k['acq_cost']:,.0f} |",
        f"| Return on ad spend | {k['roas']:.1f}x |",
        f"| Median minutes to first contact | {k['median_minutes_to_contact']:.0f} |",
        "",
        "## Key insights",
        f"1. **Speed to lead is the biggest lever.** Leads contacted within 5 minutes convert at {pct(fast)} "
        f"versus {pct(slow)} when first contact takes over 2 hours ({fast / slow:.1f}x).",
        f"2. **Source quality varies widely.** `{best}` converts at {pct(src.loc[best, 'conv_rate'])} while "
        f"`{worst}` converts at {pct(src.loc[worst, 'conv_rate'])}.",
        "3. **Agent tenure matters.** Conversion rises steadily with tenure: "
        + ", ".join(f"{b}: {pct(v)}" for b, v in t["tenure"].items()) + ".",
        f"4. **Weekends underperform.** Saturday/Sunday average {pct(wd[['Saturday', 'Sunday']].mean())} vs "
        f"weekday {pct(wd.iloc[:5].mean())}.",
        "",
        "## Recommendations",
        "- Set a 5-minute first-contact SLA with real-time alerts on uncontacted leads.",
        "- Shift paid budget towards the highest-ROAS sources and review affiliate lead quality.",
        "- Pair newer agents with experienced ones and route high-value campaigns to tenured agents.",
        "- Staff weekends with experienced agents or call back weekend leads first on Monday morning.",
        "",
        "## Source performance",
        "| Source | Leads | Conversion | Revenue (GBP) | Cost per sale (GBP) | ROAS |", "|---|---|---|---|---|---|",
    ]
    for s, r in src.iterrows():
        roas = f"{r['roas']:.1f}x" if pd.notna(r["roas"]) else "n/a (unpaid)"
        cps = f"{r['cost_per_sale']:.0f}" if r["cost"] > 0 else "0"
        lines.append(f"| {s} | {int(r['leads']):,} | {pct(r['conv_rate'])} | {r['revenue']:,.0f} | {cps} | {roas} |")
    lines += ["", "## Charts", ""] + [f"![{p.stem}](figures/{p.name})" for p in sorted(FIG.glob("*.png"))]
    (REPORTS / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    df, log = load_and_clean()
    k = kpis(df)
    t = charts_and_tables(df)
    write_report(k, t, log)
    print({key: (round(v, 4) if isinstance(v, float) else v) for key, v in k.items()})
