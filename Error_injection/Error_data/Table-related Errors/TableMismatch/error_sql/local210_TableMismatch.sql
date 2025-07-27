WITH february_orders AS (
    SELECT
        d.driver_modal AS hub_name,
        COUNT(*) AS orders_february
    FROM 
        orders o 
    LEFT JOIN
        deliveries d ON o.delivery_order_id = d.delivery_id 
    WHERE o.order_created_month = 2 AND o.order_status = 'FINISHED'
    GROUP BY
        d.driver_modal
),
march_orders AS (
    SELECT
        d.driver_modal AS hub_name,
        COUNT(*) AS orders_march
    FROM 
        orders o 
    LEFT JOIN
        deliveries d ON o.delivery_order_id = d.delivery_id 
    WHERE o.order_created_month = 3 AND o.order_status = 'FINISHED'
    GROUP BY
        d.driver_modal
)
SELECT
    fo.hub_name
FROM
    february_orders fo
LEFT JOIN 
    march_orders mo ON fo.hub_name = mo.hub_name
WHERE 
    fo.orders_february > 0 AND 
    mo.orders_march > 0 AND
    (CAST((mo.orders_march - fo.orders_february) AS REAL) / CAST(fo.orders_february AS REAL)) > 0.2;
