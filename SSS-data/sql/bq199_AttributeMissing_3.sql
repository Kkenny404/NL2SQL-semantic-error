WITH price_2020 AS (
  SELECT 
    category_name AS category, 
    AVG(state_bottle_retail / (bottle_volume_ml / 1000)) AS avg_price_liter_2020
  FROM 
    `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE 
    bottle_volume_ml > 0 
    AND EXTRACT(YEAR FROM date) = 2020
  GROUP BY 
    category
),
price_2019 AS (
  SELECT 
    category_name AS category, 
    AVG(state_bottle_retail / (bottle_volume_ml / 1000)) AS avg_price_liter_2019
  FROM 
    `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE 
    bottle_volume_ml > 0 
    AND EXTRACT(YEAR FROM date) = 2019
  GROUP BY 
    category
),
price_2021 AS (
  SELECT 
    category_name AS category, 
    AVG(state_bottle_retail / (bottle_volume_ml / 1000)) AS avg_price_liter_2021
  FROM 
    `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE 
    bottle_volume_ml > 0 
    AND EXTRACT(YEAR FROM date) = 2021
  GROUP BY 
    category
)
SELECT 
  price_2021.category, 
  price_2021.avg_price_liter_2021
FROM 
  price_2021
ORDER BY 
  price_2021.avg_price_liter_2021 DESC
LIMIT 
  10;