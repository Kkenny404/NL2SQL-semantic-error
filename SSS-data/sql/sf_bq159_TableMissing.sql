WITH
    table1 AS (
        SELECT
            "symbol",
            "avgdata" AS "data",
            "ParticipantBarcode"
        FROM (
            SELECT
                'histological_type' AS "symbol", 
                "histological_type" AS "avgdata",
                "bcr_patient_barcode" AS "ParticipantBarcode"
            FROM 
                "PANCANCER_ATLAS_1"."PANCANCER_ATLAS_FILTERED"."CLINICAL_PANCAN_PATIENT_WITH_FOLLOWUP_FILTERED"
            WHERE 
                "acronym" = 'BRCA' 
                AND "histological_type" IS NOT NULL      
        )
    ),
    table2 AS (
        SELECT
            "symbol",
            "ParticipantBarcode"
        FROM (
            SELECT
                'CDH1' AS "symbol", 
                "bcr_patient_barcode" AS "ParticipantBarcode"
            WHERE 
                "acronym" = 'BRCA'
            GROUP BY
                "ParticipantBarcode", "symbol"
        )
    ),
    summ_table AS (
        SELECT 
            n1."data" AS "data1",
            CASE 
                WHEN n2."ParticipantBarcode" IS NULL THEN 'NO' 
                ELSE 'YES' 
            END AS "data2",
            COUNT(*) AS "Nij"
        FROM
            table1 AS n1
        LEFT JOIN
            table2 AS n2 
            ON n1."ParticipantBarcode" = n2."ParticipantBarcode"
        GROUP BY
            n1."data", "data2"
    ),
    expected_table AS (
        SELECT 
            "data1", 
            "data2"
        FROM (     
            SELECT 
                "data1", 
                SUM("Nij") AS "Ni"   
            FROM 
                summ_table
            GROUP BY 
                "data1"
        ) AS Ni_table
        CROSS JOIN ( 
            SELECT 
                "data2", 
                SUM("Nij") AS "Nj"
            FROM 
                summ_table
            GROUP BY 
                "data2"
        ) AS Nj_table
        WHERE 
            Ni_table."Ni" > 10 
            AND Nj_table."Nj" > 10
    ),
    contingency_table AS (
        SELECT
            T1."data1",
            T1."data2",
            COALESCE(T2."Nij", 0) AS "Nij",
            (SUM(T2."Nij") OVER (PARTITION BY T1."data1")) * 
            (SUM(T2."Nij") OVER (PARTITION BY T1."data2")) / 
            SUM(T2."Nij") OVER () AS "E_nij"
        FROM
            expected_table AS T1
        LEFT JOIN
            summ_table AS T2
        ON 
            T1."data1" = T2."data1" 
            AND T1."data2" = T2."data2"
    )
SELECT
    SUM( ( "Nij" - "E_nij" ) * ( "Nij" - "E_nij" ) / "E_nij" ) AS "Chi2"
FROM 
    contingency_table;