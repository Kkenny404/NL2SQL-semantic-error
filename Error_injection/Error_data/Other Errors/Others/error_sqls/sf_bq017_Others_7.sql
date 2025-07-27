SELECT 
    "feature_type",
    COUNT(*) AS count
FROM 
    GEO_OPENSTREETMAP.GEO_OPENSTREETMAP.PLANET_FEATURES
GROUP BY 
    "feature_type"
ORDER BY 
    count DESC
LIMIT 5;