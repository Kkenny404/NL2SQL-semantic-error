SELECT
  genex."case_barcode" AS "case_barcode",
  genex."sample_barcode" AS "sample_barcode",
  genex."aliquot_barcode" AS "aliquot_barcode",
  genex."HGNC_gene_symbol" AS "HGNC_gene_symbol",
  clinical."Variant_Type" AS "Variant_Type",
  genex."gene_id" AS "gene_id",
  genex."normalized_count" AS "normalized_count",
  genex."project_short_name" AS "project_short_name",
  clinical."demo__gender" AS "gender",
  clinical."demo__vital_status" AS "vital_status",
  clinical."demo__days_to_death" AS "days_to_death"
FROM
  "TCGA"."TCGA_VERSIONED"."RNASEQ_HG19_GDC_2017_02" AS genex
INNER JOIN
  "TCGA"."TCGA_VERSIONED"."CLINICAL_GDC_R39" AS clinical
ON
  genex."case_barcode" = clinical."submitter_id"
WHERE
  genex."HGNC_gene_symbol" IN ('MDM2', 'TP53', 'CDKN1A','CCNE1')
  AND genex."project_short_name" = 'TCGA-BLCA'
ORDER BY
  "case_barcode",
  "HGNC_gene_symbol";