SELECT period, description, c FROM (
  SELECT 
    a.period, 
    b.description, 
    COUNT(*) c, 
    ROW_NUMBER() OVER (PARTITION BY a.period ORDER BY COUNT(*) DESC) seqnum 
  FROM `bigquery-public-data.the_met.objects` a
  JOIN (
    SELECT 
      label.description AS description, 
      object_id 
    FROM `bigquery-public-data.the_met.vision_api_data`, UNNEST(labelAnnotations) label
  ) b
  ON a.object_id = b.object_id
  LEFT JOIN `bigquery-public-data.the_met.images` i
  ON a.object_id = i.object_id
  WHERE a.period IS NOT NULL
  GROUP BY 1, 2
)
WHERE seqnum <= 3
AND c >= 500
ORDER BY period, c DESC;
