SELECT *
  FROM [NBA_Analysis].[dbo].[NBAPlayerData]

-- Make data more readable, by making it one decimal places, and making percentages xx.x%. 
SELECT "Year", Season_type, PLAYER_ID, "RANK", PLAYER, TEAM_ID, TEAM,ROUND("MIN", 1) AS "MIN",
ROUND(FGM, 1) AS FGM,
ROUND(FGA, 1) AS FGA,
ROUND(FG_PCT*100, 3) AS FG_PCT,
ROUND(FG3M, 1) AS FG3M,
ROUND(FG3A, 1) AS FG3A,
ROUND(FG3_PCT*100, 3) AS FG3_PCT,
ROUND(FTM, 1) AS FTM,
ROUND(FTA, 1) AS FTA,
ROUND(FT_PCT*100, 3) AS FT_PCT,
ROUND(OREB, 1) AS OREB,
ROUND(DREB, 1) AS DREB,
ROUND(REB, 1) AS REB,
ROUND(AST, 1) AS AST,
ROUND(STL, 1) AS STL,
ROUND(BLK, 1) AS BLK, 
ROUND(TOV, 1) AS TOV,
ROUND(PTS,1) AS PTS,
ROUND(EFF,1) AS EFF
FROM [NBA_Analysis].[dbo].[NBAPlayerData]

SELECT DISTINCT PLAYER, Season_type, "MIN", "YEAR"
FROM [NBA_Analysis].[dbo].[NBAPlayerData]
WHERE "YEAR" BETWEEN '2014-15' AND '2023-24' AND Season_type = 'Playoffs' AND "MIN" > 0;

--Main objective: see how regular season minutes affect playoff statistics compared to regular season statistics based on FGA, and FGP. 

WITH RegularSeasonStats AS (
    SELECT
        "YEAR",
		TEAM,
        PLAYER_ID,
        PLAYER,
        "MIN",
		FGA,
        FG_PCT     
    FROM [NBA_Analysis].[dbo].[NBAPlayerData]
    WHERE Season_type LIKE 'Regular%' 
),
PlayoffStats AS (
    SELECT
        "YEAR",
		TEAM,
        PLAYER_ID,
        PLAYER,
        "MIN",
		FGA,
        FG_PCT     
    FROM [NBA_Analysis].[dbo].[NBAPlayerData]
    WHERE Season_type = 'Playoffs'
)
SELECT DISTINCT
    rs."YEAR",
	rs.TEAM,
    rs.PLAYER_ID, 
    rs.PLAYER, 
    ROUND(rs."MIN", 1) AS RegularSeasonMin,
	ROUND(ps."MIN", 1) AS PlayoffsMin,
	ROUND(ps."MIN" - rs."MIN", 1) AS MIN_DIFF,
	ROUND(rs.FGA, 1) AS Regular_FGA,
	ROUND(ps.FGA, 1) AS Playoff_FGA,
	ROUND(ps.FGA - rs.FGA, 1) AS FGA_DIFF,
    ROUND(rs.FG_PCT*100, 1) AS Regular_FGP,
    ROUND(ps.FG_PCT*100, 1) AS Playoff_FGP,
	ROUND(ps.FG_PCT*100 - rs.FG_PCT*100, 1) AS FGP_DIFF

FROM RegularSeasonStats rs
LEFT JOIN PlayoffStats ps
    ON rs.PLAYER_ID = ps.PLAYER_ID 
    AND rs."YEAR" = ps."YEAR"
WHERE ps."MIN" is not NULL AND rs."MIN" >= 15
ORDER BY rs."YEAR", RegularSeasonMin DESC


