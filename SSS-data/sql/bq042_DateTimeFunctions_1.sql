SELECT
  -- Incorrectly create a timestamp using the wrong order of date components (month and day swapped).
  TIMESTAMP(CONCAT(year,"-",da,"-",mo)) AS timestamp,
  -- Replace numerical null values with actual null
  AVG(IF (temp=9999.9,
      null,
      temp)) AS temperature,
  AVG(IF (wdsp="999.9",
      null,
      CAST(wdsp AS Float64))) AS wind_speed,
  AVG(IF (prcp=99.99,
      0,
      prcp)) AS precipitation
FROM
  `bigquery-public-data.noaa_gsod.gsod20*`
WHERE
  CAST(YEAR AS INT64) > 2010
  AND CAST(YEAR AS INT64) < 2021
  AND CAST(MO AS INT64) = 6
  AND CAST(DA AS INT64) = 12
  AND stn = "725030" -- La Guardia
GROUP BY
  timestamp
ORDER BY
  timestamp ASC;