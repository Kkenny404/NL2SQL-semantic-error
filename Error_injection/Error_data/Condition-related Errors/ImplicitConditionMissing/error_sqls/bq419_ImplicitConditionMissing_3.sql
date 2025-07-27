WITH s80 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1980` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),
s81 as
(SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1981` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),
s82 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1982` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s83 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1983` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s84 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1984` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s85 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1985` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s86 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1986` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s87 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1987` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s88 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1988` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s89 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1989` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s90 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1990` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s91 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1991` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),
s92 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1992` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s93 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1993` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s94 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1994` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000),

s95 as
  (SELECT state, COUNT(event_id) as num_events
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_1995` 
  GROUP BY state 
  ORDER BY num_events DESC
  LIMIT 1000)

SELECT s80.state, 
s80.num_events + s81.num_events +  s82.num_events +  s83.num_events +  s84