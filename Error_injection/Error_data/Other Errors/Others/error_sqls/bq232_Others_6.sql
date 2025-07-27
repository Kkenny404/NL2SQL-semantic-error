WITH borough_data AS (
    SELECT 
        year, 
        month, 
        borough, 
        major_category, 
        minor_category, 
        SUM(value) AS total
    FROM 
        bigquery-public-data.london_crime.crime_by_lsoa
    GROUP BY 1,2,3,4,5
    ORDER BY 1,2
)

SELECT borough, major_category, minor_category, SUM(total) AS total_incidents
FROM borough_data
WHERE 
    year >= 2010
GROUP BY borough, major_category, minor_category
ORDER BY total_incidents DESC;