WITH AnnualSales AS (
  SELECT
    item_description,
    EXTRACT(YEAR FROM date) AS year,
    SUM(sale_dollars) AS total_sales_revenue,
    COUNT(DISTINCT invoice_and_item_number) AS unique_purchases
  FROM
    `bigquery-public-data.iowa_liquor_sales.sales`
  WHERE
    EXTRACT(YEAR FROM date) IN (2022, 2023)
    AND item_description IS NOT NULL
    AND sale_dollars IS NOT NULL
  GROUP BY
    item_description, year
)
SELECT
  item_description
FROM
  AnnualSales
WHERE
  year = 2023
ORDER BY
  total_sales_revenue DESC
LIMIT 5