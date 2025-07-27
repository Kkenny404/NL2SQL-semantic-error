WITH AvgSalaries AS (
    SELECT 
        facrank AS FacRank,
        AVG(facsalary) AS AvSalary
    FROM 
        university_faculty
    GROUP BY 
        facrank
),
SalaryDifferences AS (
    SELECT 
        university_faculty.facrank AS FacRank, 
        university_faculty.facfirstname AS FacFirstName, 
        university_faculty.faclastname AS FacLastName, 
        university_faculty.facsalary AS Salary, 
        ABS(university_faculty.facsalary - AvgSalaries.AvSalary) AS Diff
    FROM 
        university_faculty
    JOIN 
        AvgSalaries ON university_faculty.facrank = AvgSalaries.FacRank
)
SELECT 
    FacRank, 
    FacFirstName, 
    FacLastName, 
    Salary, 
    Diff
FROM 
    SalaryDifferences;