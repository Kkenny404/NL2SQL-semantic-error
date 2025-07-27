SELECT 
    STRFTIME('%Y', collision_date) AS Year
FROM 
    collisions
GROUP BY 
    Year
HAVING COUNT(case_id) > 100