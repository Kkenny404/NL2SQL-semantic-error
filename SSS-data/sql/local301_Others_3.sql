SELECT 
    SUM(sales) AS total_sales,
    COUNT(DISTINCT week_date) AS weeks_count,
    AVG(sales) AS avg_sales_per_week,
    calendar_year AS year
FROM cleaned_weekly_sales
WHERE calendar_year IN (2018, 2019, 2020)
GROUP BY calendar_year
ORDER BY year;