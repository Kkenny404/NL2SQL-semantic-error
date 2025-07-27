SELECT
  generic_name AS drug_name,
  nppes_provider_state AS state,
  ROUND(SUM(total_claim_count)) AS total_claim_count,
  ROUND(SUM(total_day_supply)) AS day_supply,
  ROUND(SUM(total_drug_cost)) / 1e6 AS total_cost_millions
FROM
  `bigquery-public-data.cms_medicare.part_d_prescriber_2014`
GROUP BY
  state,
  drug_name