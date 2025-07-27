SELECT
    year_filed,
    SUM(totfuncexpns) AS total_expenses
FROM
    `bigquery-public-data.irs_990.irs_990_2012`
WHERE
    totfuncexpns IS NOT NULL
GROUP BY
    year_filed
ORDER BY
    total_expenses ASC
LIMIT 3;