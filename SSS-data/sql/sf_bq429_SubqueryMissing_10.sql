SELECT 
  geo."state_name",
  AVG(
    (a18."median_income" - a15."median_income")
  ) AS "avg_median_income_diff",
  AVG(
    census."employed_wholesale_trade" * 0.38423645320197042 +
    census."occupation_natural_resources_construction_maintenance" * 0.48071410777129553 +
    census."employed_arts_entertainment_recreation_accommodation_food" * 0.89455676291236841 +
    census."employed_information" * 0.31315240083507306 +
    census."employed_retail_trade" * 0.51
  ) AS "avg_vulnerable"
FROM
  CENSUS_BUREAU_ACS_2.CENSUS_BUREAU_ACS."ZIP_CODES_2017_5YR" AS census
JOIN
  CENSUS_BUREAU_ACS_2.CENSUS_BUREAU_ACS."ZIP_CODES_2018_5YR" a18 ON census."geo_id" = a18."geo_id"
JOIN
  CENSUS_BUREAU_ACS_2.CENSUS_BUREAU_ACS."ZIP_CODES_2015_5YR" a15 ON census."geo_id" = a15."geo_id"
JOIN
  CENSUS_BUREAU_ACS_2.GEO_US_BOUNDARIES."ZIP_CODES" geo ON census."geo_id" = geo."zip_code"
GROUP BY geo."state_name"
ORDER BY "avg_median_income_diff" DESC
LIMIT 5;