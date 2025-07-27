WITH bea_2012 AS (
  SELECT GeoFIPS, GeoName, Earnings_per_job_avg AS earnings_2012
  FROM `bigquery-public-data.sdoh_bea_cainc30.fips`
  WHERE Year='2012-01-01' AND ENDS_WITH(GeoName, "MA") IS TRUE
),

bea_2017 AS (
  SELECT GeoFIPS, GeoName, Earnings_per_job_avg AS earnings_2017
  FROM `bigquery-public-data.sdoh_bea_cainc30.fips`
  WHERE Year='2017-01-01' AND ENDS_WITH(GeoName, "MA") IS TRUE
)

SELECT
  bea_2017.GeoFIPS, bea_2017.GeoName, bea_2017.earnings_2017, bea_2012.earnings_2012, 
  (bea_2017.earnings_2017 - bea_2012.earnings_2012) AS earnings_change
FROM bea_2017
JOIN bea_2012
ON bea_2017.GeoFIPS = bea_2012.GeoFIPS