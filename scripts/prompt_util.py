def improved_prompt(question: str, schema: str, sql: str) -> str:
    few_shot_examples = """
You are a database expert. You need to determine whether the provided SQL query is semantically correct with respect to the given natural language question and database schema.
Here are some error types you may encounter:
- Attribute-related Errors (§3.2.1): Incorrect or missing columns used in SELECT, WHERE, etc.
- Table-related Errors (§3.2.2): Incorrect or missing tables used in FROM or JOIN.
- Value-related Errors (§3.2.3): Literal values in conditions are wrong or mismatched.
- Operator-related Errors (§3.2.4): Wrong comparison operator (e.g., < instead of >=).
- Condition-related Errors (§3.2.5): Logical conditions are incomplete, missing, or incorrectly combined.
- Function-related Errors (§3.2.6): Aggregation or SQL function used is incorrect or missing.
- Clause-related Errors (§3.2.7): Missing or misused clauses such as GROUP BY, ORDER BY, LIMIT.
- Subquery-related Errors (§3.2.8): Incorrect subquery logic or structure.
- Other Errors (§3.2.9): Any semantic mismatch not covered above.
Also, you can learn the following examples:
Example 1:
question: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.
schema: 
sql: SELECT T2.`Free Meal Count (Ages 5-17)` / T2.`Enrollment (Ages 5-17)` AS EligibleFreeRate FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.EdOpsName = 'Continuation School' AND T2.`Enrollment (Ages 5-17)` > 0 AND T2.`Free Meal Count (Ages 5-17)` IS NOT NULL AND T2.`Enrollment (Ages 5-17)` IS NOT NULL ORDER BY EligibleFreeRate ASC LIMIT 3
equivalant: true,
error_types: []

Example 2:
question: Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.
evidence: Charter schools refers to `Charter School (Y/N)` = 1 in the frpm
db_id: california_schools
sql: SELECT phone FROM schools WHERE charter = 1 AND opendate > '2000-01-01'
equivalant: false,
error_types: [
    error_type: Table-Related Errors
    sub_error_type: Table Missing

    error_type: Condition-Related Errors
    sub_error_type": Explicit Condition Missing

    error_type: Attribute-Related Errors
    sub_error_type: Attribute Mismatch ]

Example 3:
question": "What is the administrator's email address for the school with the highest number of test takers who received SAT scores of at least 1500?Provide the name of the school.",
evidence: "",
db_id: california_schools",
sql": select schools.admemail1 , schools.school from schools inner join satscores on schools.cdscode = satscores.cds where satscores.numge1500 group by schools.admemail1 , schools.school order by count(satscores.numge1500) desc limit 1",
equivalant": false,
error_types": [
    error_type: Condition-Related Errors,
    sub_error_type": Explicit Condition Missing

    error_type": Clause-Related Errors
    sub_error_type: Clause Redundancy

    error_type: Other Errors
    sub_error_type: ASC/DESC
    ]
"""

    task_prompt = f"""
Now evaluate the following:

NL: {question}
Schema: {schema}
SQL: {sql}

Please follow this format:
- Initial Decision: [Yes / No]
- Reasoning:
- Reflection:
- Final Decision:
- Final Explanation:
"""

    return few_shot_examples.strip() + "\n\n" + task_prompt.strip()
