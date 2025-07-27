SELECT
    TO_CHAR(TO_TIMESTAMP(a."created_at" / 1000000.0), 'YYYY-MM') AS "month",
    c."category",
    SUM(b."sale_price") AS "TPV",
    SUM(c."cost") AS "total_cost",
    COUNT(DISTINCT a."order_id") AS "TPO",
    SUM(b."sale_price" - c."cost") AS "total_profit",
    SUM((b."sale_price" - c."cost") / c."cost") AS "Profit_to_cost_ratio"
FROM 
    "THELOOK_ECOMMERCE"."THELOOK_ECOMMERCE"."ORDERS" AS a
JOIN 
    "THELOOK_ECOMMERCE"."THELOOK_ECOMMERCE"."ORDER_ITEMS" AS b
    ON a."order_id" = b."order_id"
JOIN 
    "THELOOK_ECOMMERCE"."THELOOK_ECOMMERCE"."PRODUCTS" AS c
    ON b."product_id" = c."id"
WHERE 
    a."status" = 'Complete'
    AND TO_TIMESTAMP(a."created_at" / 1000000.0) BETWEEN TO_TIMESTAMP('2023-01-01') AND TO_TIMESTAMP('2023-12-31')
    AND c."category" = 'Sleep & Lounge'
GROUP BY
    TO_CHAR(TO_TIMESTAMP(a."created_at" / 1000000.0), 'YYYY-MM'),
    c."category"
ORDER BY
    "month";