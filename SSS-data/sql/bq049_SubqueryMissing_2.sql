WITH DUBUQUE_LIQUOR_CTE AS (
SELECT
  CASE
      WHEN UPPER(category_name) LIKE 'BUTTERSCOTCH SCHNAPPS' THEN 'All Other'
      WHEN UPPER(category_name) LIKE '%WHISKIES' 
            AND UPPER(category_name) NOT LIKE '%RYE%'
            AND UPPER(category_name) NOT LIKE '%BOURBON%'
            AND UPPER(category_name) NOT LIKE '%SCOTCH%'     THEN 'Other Whiskey'
      WHEN UPPER(category_name) LIKE '%RYE%'                 THEN 'Rye Whiskey'
      WHEN UPPER(category_name) LIKE '%BOURBON%'             THEN 'Bourbon Whiskey'
      WHEN UPPER(category_name) LIKE '%SCOTCH%'              THEN 'Scotch Whiskey'
      ELSE 'All Other'
  END                              AS category_group,
  EXTRACT(MONTH FROM date)         AS month,
  LEFT(CAST(zip_code AS string),5) AS zip_code,
  ROUND(SUM(sale_dollars), 2)      AS sale_dollars_sum

FROM 
  bigquery-public-data.iowa_liquor_sales.sales

WHERE
  UPPER(county)               = 'DUBUQUE'
  AND EXTRACT(YEAR FROM date) = 2022

GROUP BY
  category_group,
  month,
  zip_code
  
ORDER BY 
  category_group,
  month,
  zip_code
),

DUBUQUE_POPULATION_CTE AS (
SELECT
  zipcode,
  SUM(population) AS population_sum
FROM bigquery-public-data.census_bureau_usa.population_by_zip_2010
WHERE 
  minimum_age >= 21
GROUP BY 
  zipcode
),
MONTH_INFO AS (
SELECT 
  l.month,
  l.zip_code,
  l.sale_dollars_sum,
  ROUND(sale_dollars_sum/p.population_sum, 2) AS dollars_per_capita
FROM 
  DUBUQUE_LIQUOR_CTE AS l
  LEFT JOIN 
  DUBUQUE_POPULATION_CTE AS p
  ON l.zip_code = p.zipcode
WHERE
  category_group = 'Bourbon Whiskey'
GROUP BY 
  category_group,
  zip_code,
  month,
  sale_dollars_sum,
  zipcode,
  population_sum
ORDER BY
  zip_code,
  month
)
SELECT
    t.month,
    t.zip_code,
    t.dollars_per_capita
FROM MONTH_INFO t
ORDER BY t.zip_code, t.month;