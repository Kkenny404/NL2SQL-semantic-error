WITH TractPop AS (
    SELECT
        CG."BlockGroupID",
        FCV."CensusValue",
        CG."StateCountyTractID",
        CG."BlockGroupPolygon"
    FROM
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."LU_GeographyExpanded" CG
    JOIN
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Fact_CensusValues_ACS2021_ByZip" FCV
        ON CG."ZipCode" = FCV."ZipCode"
    WHERE
        CG."PreferredStateAbbrev" = 'NY'
        AND FCV."MetricID" = 'B01003_001E'
),

TractGroup AS (
    SELECT
        CG."ZipCode",
        SUM(FCV."CensusValueByZip") AS "TotalTractPop"
    FROM
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."LU_GeographyExpanded" CG
    JOIN
        CENSUS_GALAXY__ZIP_CODE_TO_BLOCK_GROUP_SAMPLE.PUBLIC."Fact_CensusValues_ACS2021_ByZip" FCV
        ON CG."ZipCode" = FCV."ZipCode"
    WHERE
        CG."PreferredStateAbbrev" = 'NY'
        AND FCV."MetricID" = 'B01003_001E'
    GROUP BY
        CG."ZipCode"
)

SELECT
    TP."BlockGroupID",
    TP."CensusValue",
    TP."StateCountyTractID",
    TG."TotalTractPop",
    CASE WHEN TG."TotalTractPop" <> 0 THEN TP."CensusValue" / TG."TotalTractPop" ELSE 0 END AS "BlockGroupRatio"
FROM
    TractPop TP
JOIN
    TractGroup TG
    ON TP."StateCountyTractID" = TG."ZipCode";
