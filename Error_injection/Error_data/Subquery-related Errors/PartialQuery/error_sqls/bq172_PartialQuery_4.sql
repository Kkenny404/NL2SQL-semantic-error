SELECT
  drug_name AS drug_name,
  ROUND(SUM(total_claim_count)) AS total_claim_count
FROM
  `bigquery-public-data.cms_medicare.part_d_prescriber_2014`
WHERE
  nppes_provider_state = 'NY'
GROUP BY
  drug_name
ORDER BY
  total_claim_count DESC
LIMIT 1