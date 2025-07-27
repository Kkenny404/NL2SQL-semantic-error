WITH russia_Data AS (
  SELECT DISTINCT 
    id.country_name,
    id.value,
    id.indicator_name
  FROM (
    SELECT
      country_code,
      region
    FROM
      bigquery-public-data.world_bank_intl_debt.country_summary
    WHERE
      region != "" 
  ) cs
  INNER JOIN (
    SELECT
      country_code,
      country_name,
      value, 
      indicator_name
    FROM
      bigquery-public-data.world_bank_intl_debt.international_debt
    WHERE
      country_code = 'RUS'
  ) id
  ON
    cs.country_code = id.country_code
  WHERE value IS NOT NULL
)
-- Only returns the filtered rows, not the count
SELECT 
  *
FROM 
  russia_Data
WHERE 
  value = 0;