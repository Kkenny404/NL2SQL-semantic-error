WITH UserPairUpvotes AS (
  SELECT
    ToUsers."UserName" AS "ToUserName",
    FromUsers."UserName" AS "FromUserName",
    COUNT(DISTINCT "KernelVotes"."Id") AS "UpvoteCount"
  FROM META_KAGGLE.META_KAGGLE.KERNELVOTES AS "KernelVotes"
  INNER JOIN META_KAGGLE.META_KAGGLE.USERS AS FromUsers
    ON FromUsers."Id" = "KernelVotes"."UserId"
  INNER JOIN META_KAGGLE.META_KAGGLE.USERS AS ToUsers
    ON ToUsers."Id" = "KernelVotes"."UserId"
  GROUP BY
    ToUsers."UserName",
    FromUsers."UserName"
),
TopPairs AS (
  SELECT
    "ToUserName",
    "FromUserName",
    "UpvoteCount",
    ROW_NUMBER() OVER (ORDER BY "UpvoteCount" DESC) AS "Rank"
  FROM UserPairUpvotes
),
ReciprocalUpvotes AS (
  SELECT
    t."ToUserName",
    t."FromUserName",
    t."UpvoteCount" AS "UpvotesReceived",
    COALESCE(u."UpvoteCount", 0) AS "UpvotesGiven"
  FROM TopPairs t
  LEFT JOIN UserPairUpvotes u
    ON t."ToUserName" = u."FromUserName" AND t."FromUserName" = u."ToUserName"
  WHERE t."Rank" = 1
)
SELECT
  "ToUserName" AS "UpvotedUserName",
  "FromUserName" AS "UpvotingUserName",
  "UpvotesReceived" AS "UpvotesReceivedByUpvotedUser",
  "UpvotesGiven" AS "UpvotesGivenByUpvotedUser"
FROM ReciprocalUpvotes
ORDER BY "UpvotesReceived" DESC, "UpvotesGiven" DESC;
