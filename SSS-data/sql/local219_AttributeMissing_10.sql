WITH match_view AS(
SELECT
    M.id,
    L.name AS league,
    M.season,
    M.match_api_id,
    T.team_long_name AS home_team,
    TM.team_long_name AS away_team,
    M.home_team_goal,
    M.away_team_goal
FROM
    match M
LEFT JOIN
    league L ON M.league_id = L.id
LEFT JOIN
    team T ON M.home_team_api_id = T.team_api_id
LEFT JOIN
    team TM ON M.away_team_api_id = TM.team_api_id
),
match_score AS
(
    SELECT  -- Displaying teams and their goals as home_team
        id,
        home_team AS team,
        CASE
            WHEN home_team_goal > away_team_goal THEN 1 ELSE 0 END AS Winning_match
    FROM
        match_view

    UNION ALL

    SELECT  -- Displaying teams and their goals as away_team
        id,
        away_team AS team,
        CASE
            WHEN away_team_goal > home_team_goal THEN 1 ELSE 0 END AS Winning_match
    FROM
        match_view
),
winning_matches AS
(
    SELECT  -- Displaying total match wins for each team
        MV.league,
        M.team,
        COUNT(CASE WHEN M.Winning_match = 1 THEN 1 END) AS wins,
        ROW_NUMBER() OVER(PARTITION BY MV.league ORDER BY COUNT(CASE WHEN M.Winning_match = 1 THEN 1 END) ASC) AS rn
    FROM
        match_score M
    JOIN
        match_view MV
    ON
        M.id = MV.id
    GROUP BY
        MV.league,
        team
    ORDER BY
        league,
        wins ASC
)
SELECT
    league,
    team
FROM
    winning_matches
WHERE
    rn = 1  -- Getting the team with the least number of wins in each league
ORDER BY
    league;