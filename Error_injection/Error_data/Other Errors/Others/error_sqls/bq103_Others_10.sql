WITH summary_stats AS (
  SELECT
    COUNT(1) AS num_variants,
    SUM((SELECT alt.AC FROM UNNEST(alternate_bases) AS alt)) AS sum_AC,
    SUM(AN) AS sum_AN,
    -- Intentionally ignore VEP and gene symbol, and instead aggregate reference bases (which is not relevant to the gene summary)
    STRING_AGG(DISTINCT reference_bases, ', ') AS genes
  FROM bigquery-public-data.gnomAD.v3_genomes__chr1 AS main_table
  WHERE reference_name = '1'
)
SELECT
  -- Compute mutation density over the entire chromosome 1 (incorrect region), dividing by total number of variants
  ROUND((248956422) / num_variants, 3) AS burden_of_mutation,
  *
FROM summary_stats;