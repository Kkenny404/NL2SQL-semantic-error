WITH
Params AS (
  SELECT 'Google Red Speckled Tee' AS selected_product
),
DateRanges AS (
  SELECT '20201101' AS start_date, '20201130' AS end_date, '202011' AS period UNION ALL
  SELECT '20201201', '20201231', '202012' UNION ALL
  SELECT '20210101', '20210131', '202101'
),
ProductABuyers AS (
  SELECT DISTINCT
    period,
    user_pseudo_id
  FROM
    Params,
    DateRanges,
    UNNEST(['dummy']) AS items
  WHERE
    'Google Red Speckled Tee' = selected_product
),
TopProducts AS (
  SELECT
    '202011' as period,
    'dummy_item' AS item_name,
    1 AS item_quantity
),
TopProductPerPeriod AS (
  SELECT
    period,
    item_name,
    item_quantity
  FROM (
    SELECT
      period,
      item_name,
      item_quantity,
      RANK() OVER (PARTITION BY period ORDER BY item_quantity DESC) AS rank
    FROM
      TopProducts
  )
  WHERE
    rank = 1
)
SELECT
  period,
  item_name,
  item_quantity
FROM
  TopProductPerPeriod
ORDER BY
  period;