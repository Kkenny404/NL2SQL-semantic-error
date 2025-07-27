SELECT
    borough,
    major_category,
    SUM(value) AS no_of_incidents
FROM
    `bigquery-public-data.london_crime.crime_by_lsoa`
WHERE
    borough = 'Barking and Dagenham'
GROUP BY
    borough,
    major_category
ORDER BY
    no_of_incidents DESC;