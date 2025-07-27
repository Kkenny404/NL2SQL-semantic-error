WITH summary_stats AS (
  SELECT
    COUNT(1) AS num_variants,
    SUM((SELECT alt.AC FROM UNNEST(alternate_bases) AS alt)) AS sum_AC,
    SUM(AN) AS sum_AN
    -- Missing: genes (distinct gene symbols from VEP)
  FROM bigquery-public-data.gnomAD.v3_genomes__chr1 AS main_table
  WHERE start_position >= 55039447 AND start_position <= 55064852
)
SELECT
  ROUND((55064852 - 55039447) / num_variants, 3) AS burden_of_mutation,
  *
FROM summary_stats;