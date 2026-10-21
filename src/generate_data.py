"""Generate a realistic *synthetic* lead/sales dataset for a multi-site call-center operation.

No real customer data is used. The generator bakes in a few patterns (source quality,
speed-to-lead effect, agent tenure, weekday effects) so the analysis has something to find.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = Path(__file__).resolve().parents[1] / "data"

SITES = {"UK": 0.30, "Pakistan": 0.35, "India": 0.20, "South Africa": 0.15}
SOURCES = {"google_ads": (0.30, 0.16), "facebook": (0.35, 0.09), "referral": (0.10, 0.28),
           "organic": (0.15, 0.18), "affiliate": (0.10, 0.06)}
CAMPAIGNS = ["debt_advice", "car_finance_claims", "rental_claims"]
CAMPAIGN_VALUE = {"debt_advice": 420, "car_finance_claims": 650, "rental_claims": 380}


def make_agents(n: int = 40) -> pd.DataFrame:
    sites = RNG.choice(list(SITES), size=n, p=list(SITES.values()))
    tenure = RNG.integers(1, 48, size=n)  # months
    return pd.DataFrame({"agent_id": [f"A{i:03d}" for i in range(1, n + 1)], "site": sites, "tenure_months": tenure})


def make_leads(agents: pd.DataFrame, n: int = 25_000) -> pd.DataFrame:
    created = pd.Timestamp("2026-01-01") + pd.to_timedelta(RNG.integers(0, 273 * 24 * 60, size=n), unit="m")
    src_names = list(SOURCES)
    src = RNG.choice(src_names, size=n, p=[SOURCES[s][0] for s in src_names])
    campaign = RNG.choice(CAMPAIGNS, size=n, p=[0.5, 0.3, 0.2])
    agent = agents.sample(n=n, replace=True, random_state=7).reset_index(drop=True)
    # minutes until first contact: log-normal, long tail
    first_contact = np.round(RNG.lognormal(mean=3.0, sigma=1.1, size=n)).astype(int)

    base = np.array([SOURCES[s][1] for s in src])
    speed_factor = np.where(first_contact <= 5, 1.6, np.where(first_contact <= 30, 1.15, np.where(first_contact <= 120, 0.85, 0.55)))
    tenure_factor = 0.8 + np.minimum(agent["tenure_months"].to_numpy(), 24) / 60
    weekday = created.dayofweek.to_numpy()
    weekday_factor = np.where(weekday >= 5, 0.8, 1.0)
    p_convert = np.clip(base * speed_factor * tenure_factor * weekday_factor, 0, 0.9)
    converted = RNG.random(n) < p_convert

    contacted = converted | (RNG.random(n) < 0.78)
    qualified = converted | (contacted & (RNG.random(n) < 0.42))
    value = np.where(converted, np.round([CAMPAIGN_VALUE[c] for c in campaign] * RNG.normal(1, 0.15, n), 2), 0.0)
    cost = np.round(np.select([src == "google_ads", src == "facebook", src == "affiliate"], [14.0, 7.5, 22.0], 0.0) * RNG.normal(1, 0.1, n), 2)

    df = pd.DataFrame({
        "lead_id": np.arange(1, n + 1),
        "created_at": created,
        "source": src,
        "campaign": campaign,
        "agent_id": agent["agent_id"],
        "site": agent["site"],
        "minutes_to_first_contact": first_contact,
        "contacted": contacted,
        "qualified": qualified,
        "converted": converted,
        "revenue_gbp": value,
        "acquisition_cost_gbp": cost,
    })
    # inject some messy data to clean, like real exports
    dup = df.sample(150, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(80, random_state=2).index, "source"] = df["source"].str.upper()
    df.loc[df.sample(60, random_state=3).index, "minutes_to_first_contact"] = -1
    return df.sample(frac=1, random_state=4).reset_index(drop=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    agents = make_agents()
    leads = make_leads(agents)
    agents.to_csv(OUT / "agents.csv", index=False)
    leads.to_csv(OUT / "leads_raw.csv", index=False)
    print(f"wrote {len(leads):,} lead rows and {len(agents)} agents to {OUT}")
