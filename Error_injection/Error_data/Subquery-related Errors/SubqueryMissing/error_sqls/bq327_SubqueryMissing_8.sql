-- Subquery missing: The filtering of aggregated countries via the country_summary subquery is omitted
SELECT 
  COUNT(*) AS number_of_indicators_with_zero
FROM 
  bigquery-public-data.world_bank_intl_debt.international_debt
WHERE 
  country_code = 'RUS'
  AND value IS NOT NULL
  AND value = 0;