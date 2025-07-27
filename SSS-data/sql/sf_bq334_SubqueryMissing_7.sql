WITH all_transactions AS (
    SELECT 
        TO_TIMESTAMP_NTZ("block_timestamp" / 1000000) AS "timestamp",
        "value",
        'input' AS "type"
    FROM 
        "CRYPTO"."CRYPTO_BITCOIN"."INPUTS"
    UNION ALL
    SELECT 
        TO_TIMESTAMP_NTZ("block_timestamp" / 1000000) AS "timestamp",
        "value",
        'output' AS "type"
    FROM 
        "CRYPTO"."CRYPTO_BITCOIN"."OUTPUTS"
),
-- MISSING filtered_transactions subquery!
average_output_values AS (
    SELECT
        EXTRACT(YEAR FROM "timestamp") AS "year",
        AVG("value") AS "avg_value"
    FROM 
        all_transactions
    WHERE "type" = 'output'
    GROUP BY "year"
),
average_transaction_values AS (
    SELECT 
        EXTRACT(YEAR FROM TO_TIMESTAMP_NTZ("block_timestamp" / 1000000)) AS "year",
        AVG("output_value") AS "avg_transaction_value" 
    FROM 
        "CRYPTO"."CRYPTO_BITCOIN"."TRANSACTIONS" 
    GROUP BY "year" 
    ORDER BY "year"
),
common_years AS (
    SELECT
        ao."year",
        ao."avg_value" AS "avg_output_value",
        atv."avg_transaction_value"
    FROM
        average_output_values ao
    JOIN
        average_transaction_values atv 
        ON ao."year" = atv."year"
)

SELECT
    "year",
    "avg_transaction_value" - "avg_output_value" AS "difference"
FROM
    common_years
ORDER BY
    "year";