import re

sql_query = """
SELECT *
FROM (
  SELECT
    `start`,
    `end`,
    ROUND(
      POW(ABS(case_ref_count - (ref_count / allele_count) * case_count) - 0.5, 2) / ((ref_count / allele_count) * case_count) +
      POW(ABS(control_ref_count - (ref_count / allele_count) * control_count) - 0.5, 2) / ((ref_count / allele_count) * control_count) +
      POW(ABS(case_alt_count - (alt_count / allele_count) * case_count) - 0.5, 2) / ((alt_count / allele_count) * case_count) +
      POW(ABS(control_alt_count - (alt_count / allele_count) * control_count) - 0.5, 2) / ((alt_count / allele_count) * control_count),
      3
    ) AS chi_squared_score
  FROM (
    SELECT
      reference_name,
      `start`,
      `end`,
      reference_bases,
      alternate_bases,
      vt,
      SUM(ref_count + alt_count) AS allele_count,
      SUM(ref_count) AS ref_count,
      SUM(alt_count) AS alt_count,
      SUM(IF(is_case, CAST(ref_count + alt_count AS INT64), 0)) AS case_count,
      SUM(IF(NOT is_case, CAST(ref_count + alt_count AS INT64), 0)) AS control_count,
      SUM(IF(is_case, ref_count, 0)) AS case_ref_count,
      SUM(IF(is_case, alt_count, 0)) AS case_alt_count,
      SUM(IF(NOT is_case, ref_count, 0)) AS control_ref_count,
      SUM(IF(NOT is_case, alt_count, 0)) AS control_alt_count
    FROM (
      SELECT
        v.reference_name,
        v.`start`,
        v.`end`,
        v.reference_bases,
        v.alternate_bases,
        v.vt,
        ('EAS' = p.super_population) AS is_case,
        IF(call.genotype[SAFE_OFFSET(0)] = 0, 1, 0) AS ref_count,
        IF(call.genotype[SAFE_OFFSET(0)] = 1, 1, 0) AS alt_count
      FROM
        `spider2-public-data.1000_genomes.variants` AS v,
        UNNEST(v.call) AS call
      JOIN
        `spider2-public-data.1000_genomes.sample_info` AS p
      ON
        call.call_set_name = p.sample
      WHERE
        v.reference_name = '12'
    )
    GROUP BY
      reference_name,
      `start`,
      `end`,
      reference_bases,
      alternate_bases,
      vt
  )
  WHERE
    (ref_count / allele_count) * case_count >= 5.0
    AND (ref_count / allele_count) * control_count >= 5.0
    AND (alt_count / allele_count) * case_count >= 5.0
    AND (alt_count / allele_count) * control_count >= 5.0
)
WHERE
  chi_squared_score >= 29.71679
ORDER BY
  chi_squared_score DESC
"""


bird_sql = """SELECT T2.PostId FROM users AS T1 INNER JOIN postHistory AS T2 ON T1.Id = T2.UserId INNER JOIN posts AS T3 ON T2.PostId = T3.Id WHERE T1.DisplayName = 'slashnick' ORDER BY T3.AnswerCount DESC LIMIT 1	codebase_community
"""
# 使用正则表达式提取所有 token（包括符号、关键词、字段等）
tokens = re.findall(r"\b\w+\b|[^\s\w]", bird_sql)
lines = [line for line in bird_sql.splitlines() if line.strip()]

print(f"Token 数: {len(tokens)}")
print(f"行数（非空）: {len(lines)}")
