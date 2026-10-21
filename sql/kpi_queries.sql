-- KPI queries used to cross-check the Pandas analysis.
-- Runs on SQLite (see src/run_sql.py) and is standard enough for PostgreSQL/MySQL.

-- 1. Funnel and conversion rate by source
SELECT source,
       COUNT(*)                                             AS leads,
       ROUND(AVG(contacted) * 100, 1)                       AS contact_rate_pct,
       ROUND(AVG(converted) * 100, 1)                       AS conversion_rate_pct,
       ROUND(SUM(revenue_gbp), 0)                           AS revenue_gbp,
       ROUND(SUM(acquisition_cost_gbp) / NULLIF(SUM(converted), 0), 0) AS cost_per_sale_gbp
FROM leads
GROUP BY source
ORDER BY conversion_rate_pct DESC;

-- 2. Speed-to-lead buckets
SELECT CASE
         WHEN minutes_to_first_contact <= 5   THEN '1. <=5 min'
         WHEN minutes_to_first_contact <= 30  THEN '2. 6-30 min'
         WHEN minutes_to_first_contact <= 120 THEN '3. 31-120 min'
         ELSE '4. >2 hours'
       END                                 AS speed_bucket,
       COUNT(*)                            AS leads,
       ROUND(AVG(converted) * 100, 1)      AS conversion_rate_pct
FROM leads
WHERE minutes_to_first_contact IS NOT NULL
GROUP BY speed_bucket
ORDER BY speed_bucket;

-- 3. Top 5 agents by revenue, with their rank inside their site (window function)
SELECT *
FROM (
  SELECT l.agent_id,
         l.site,
         a.tenure_months,
         COUNT(*)                                  AS leads,
         ROUND(AVG(l.converted) * 100, 1)          AS conversion_rate_pct,
         ROUND(SUM(l.revenue_gbp), 0)              AS revenue_gbp,
         RANK() OVER (PARTITION BY l.site ORDER BY SUM(l.revenue_gbp) DESC) AS rank_in_site
  FROM leads l
  JOIN agents a ON a.agent_id = l.agent_id
  GROUP BY l.agent_id, l.site, a.tenure_months
)
ORDER BY revenue_gbp DESC
LIMIT 5;

-- 4. Month-over-month revenue growth (CTE + LAG)
WITH monthly AS (
  SELECT strftime('%Y-%m', created_at) AS month,
         SUM(revenue_gbp)              AS revenue
  FROM leads
  GROUP BY month
)
SELECT month,
       ROUND(revenue, 0) AS revenue_gbp,
       ROUND((revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;
