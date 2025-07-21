✅
- local197_TableMismatch.sql - line 8

        rental AS pm  -- ❌ Table Mismatch: 'payment' replaced with incorrect 'rental' table

- bq268_TableMismatch.sql

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` --> FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160916`

- local309_TableMismatch.sql

        drivers -->  drivers_ext；
        constructors -->  constructors_ext；

- sf011_TableMismatch.sql

    Modifications Introduced:

        ✅ Replaced Dim_CensusGeography --> LU_GeographyExpanded

        ✅ Replaced Fact_CensusValues_ACS2021 --> Fact_CensusValues_ACS2021_ByZip

        ✅ Changed join keys to ZipCode, which, although syntactically valid, are semantically unrelated to the original intent involving Block Group and Tract level granularity.

    These changes preserve SQL validity while introducing a semantic misalignment between the question’s intent (population ratios by block group and census tract) and the actual SQL logic (which incorrectly operates at the ZIP code level). This constitutes a classic Table Mismatch error.

- sf_bq167_TableMismatch.sql

        Replaced FORUMMESSAGEVOTES with KERNELVOTES
    Semantics are incorrect because the query now counts kernel votes instead of forum message upvotes
    
- local210_TableMismatch.sql

        The original query joins orders → stores → hubs to analyze order growth by hub.

        The incorrect version instead joins orders → deliveries and uses driver_modal from the drivers/deliveries path as a misidentified "hub", which does not represent actual physical hubs.

- bq374_TableMismatch.sql
    
        In the `filtered_data` CTE, replaced the original table `google_analytics_sample.*` with `google_analytics_sample.ga_sessions_intraday_*`.

        This table define as only current session data and lacks the historical session records needed to determine future transactions. This causes a semantic drift.

-  bq280_TableMismatch.sql

        The only semantic error injected is replacing ‘posts_answers’ (correct table for answers) with ‘posts_tag_wiki’, which is unrelated to user answers.
        The query now counts entries from a table that does not represent answers, despite having similar fields.

---

- sf_bq334_TableRedundancy.sql

        -- Redundant table join injected below
        block_info AS (
            SELECT 
                number AS block_number,
                timestamp AS block_time
            FROM 
                "CRYPTO"."CRYPTO_BITCOIN"."BLOCKS"
        ),
    An unnecessary table was included

- bq144_TableRedundancy.sql

        -- Redundant table ‘mbb_teams’ join injected below
        LEFT JOIN `data-to-insights.ncaa.mbb_teams` AS teams

- bq419_TableRedundancy.sql

    -- Redundant table 'ncei_nrs_pad_metadata' join injected below
        
        CROSS JOIN `bigquery-public-data.noaa_passive_bioacoustic.ncei_nrs_pad_metadata`
    This table is about passive bioacoustic metadata and is completely irrelevant to querying storm events.

- sf_bq246_TableRedundancy.sql

        -- Redundant table 'BOTANIC'

        LEFT JOIN PATENTSVIEW.PATENTSVIEW.BOTANIC AS bot ON app."patent_id" = bot."patent_id"

- ga020_TableRedundancy.sql

    -- Redundant CTE 

        analytics_data AS (
            SELECT COUNT(*) as analytics_count
            FROM `firebase-public-project.analytics_153293282.events_20180801`
            )
    -- Redundant JOIN

        CROSS JOIN analytics_data
    The 'events_20180801' table is already included by the wildcard 'events_*' in the original query.
    This CTE only counts the number of rows in the table, but this information is not used in the query at all.
    It has nothing to do with user retention analysis

- bq354_TableRedundancy.sql

    -- line 69-72

        LEFT JOIN
        `bigquery-public-data.cms_synthetic_patient_data_omop.drug_era` de ON co.person_id = de.person_id  -- Redundant join; not used

- ga008_TableRedundancy.sql
    -- line 17-18

        LEFT JOIN `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20201115` AS redundant_table ON UserInfo.user_pseudo_id = redundant_table.user_pseudo_id

    Because the data of this table has been included by the events_* wildcard and WHERE _TABLE_SUFFIX BETWEEN '20201101' AND '20201130' condition in the original query

- bq366_TableRedundancy.sql

    -- line 15 -16

        LEFT JOIN `bigquery-public-data.the_met.images` i  -- Redundant join
        ON a.object_id = i.object_id

- bq039_TableRedundancy.sql

    -- line 20-21
    
        LEFT JOIN `bigquery-public-data.new_york_subway.stations` s
        join ON tz.zone_name = s.borough_name

    Unrelated to the taxi trip query and never used in filters, calculations, or output.


- sf_bq252_TableRedundancy.sql
    -- line 25-26

        LEFT JOIN GITHUB_REPOS.GITHUB_REPOS.SAMPLE_COMMITS AS sc
        ON f."repo_name" = sc."repo_name"


---

- sf_bq264_TableMissing.sql

    --line 46-47 delete table "THELOOK_ECOMMERCE"."THELOOK_ECOMMERCE"."USERS"


- sf_bq159_TableMissing.sql

    -- line 28 delete necessary tale “MC3_MAF_V5_ONE_PER_TUMOR_SAMPLE”


- sf_bq150_TableMissing.sql

    -- line 19 delete "SOMATIC_MUTATION_MC3" from the cohortVar CTE


- ga002_TableMissing_prompt.txt

    -- line 10 - 22, removing table using 'bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*'


- bq398_TableMissing.sql

    -- remove ‘country_summary’

- bq003_TableMissing.sql

    -- remove ‘UNNEST (hits.product)’
    -- Since the query specifically needs product revenue data from the hits.product array to classify sessions properly.


- local199_TableMissing.sql

    -- remove table 'STAFF'

- bq397_TableMissing.sql

    -- remove ’tmp1‘

- sf_bq412_TableMissing.sql

    -- remove "GOOGLE_ADS"




































---
❌
- sf_bq219_TableMismatch_prompt.txt
*Cancel

- ga018_TableMismatch_prompt.txt
*cancel