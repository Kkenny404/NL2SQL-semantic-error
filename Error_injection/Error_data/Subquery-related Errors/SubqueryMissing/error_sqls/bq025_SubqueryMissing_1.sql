SELECT
  age.country_name,
  SUM(age.population) AS under_25,
  pop.midyear_population AS total,
  ROUND((SUM(age.population) / pop.midyear_population) * 100,2) AS pct_under_25
FROM
  `bigquery-public-data.census_bureau_international.midyear_population_agespecific` age
INNER JOIN
  `bigquery-public-data.census_bureau_international.midyear_population` pop
ON
  age.country_code = pop.country_code
WHERE
  age.year = 2020
  AND pop.year = 2020
  AND age.age < 20
GROUP BY
  age.country_name,
  pop.midyear_population
ORDER BY
  pct_under_25 DESC
LIMIT 10