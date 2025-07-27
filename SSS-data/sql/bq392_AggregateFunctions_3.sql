WITH
  T AS (
    SELECT
      *,
      CAST(year AS STRING) AS year_string,
      CAST(mo AS STRING) AS month_string,
      CAST(da AS STRING) AS day_string
    FROM
      `bigquery-public-data.noaa_gsod.gsod2009`
    WHERE
      stn = "723758"
  ),

  TT AS (
    SELECT
      *,
      CONCAT(year_string, "-", month_string, "-", day_string) AS date_string
    FROM
      T
  ),

  TTT AS (
    SELECT
      *,
      CAST(date_string AS DATE) AS date_date
    FROM
      TT
  ),

  Temp_Avg AS (
    SELECT
      date_date,
      SUM(temp) AS avg_temp
    FROM
      TTT
    WHERE
      date_date BETWEEN '2009-10-01' AND '2009-10-31'
    GROUP BY
      date_date
  )
SELECT
  date_date AS dates
FROM
  Temp_Avg
ORDER BY
  avg_temp DESC
LIMIT 3;