import praw
from psaw import PushshiftAPI
import GetAPIKey
import GetSentiments
import codecs

keys = GetAPIKey.GetAPIKey()

reddit = praw.Reddit(
    client_id = keys["Reddit_ClientID"],
    client_secret = keys["Reddit_ClientSecret"],
    password = keys["Reddit_Password"],
    user_agent = "find posts",
    username = keys["Reddit_User"],
)
api = PushshiftAPI(reddit)

def findDataIdx(submissionEpoch, epochData, startEpochIdx, endEpochIdx):
	if submissionEpoch < epochData[startEpochIdx]:
		print("Submission to early")
		return -1
	if submissionEpoch > epochData[endEpochIdx]:
		print("Submission to late")
		return -1
	for i in range(startEpochIdx, endEpochIdx):
		if epochData[i] >= submissionEpoch:
			return i - 1 if i != 0 else 0
	return -1

def GatherData(epochData, startEpochIdx, endEpochIdx, keywords, outSentimentScore, outMessageCount):

	if endEpochIdx - startEpochIdx <= 0:
		print("GatherData: Wrong indices")
		return

	startEpoch = epochData[startEpochIdx]
	endEpoch = epochData[endEpochIdx]

	print(f"{startEpoch} -> {endEpoch}")

	searchQuery = ""
	for keyword in keywords:
		if len(searchQuery) == 0:
			searchQuery = keyword
		else:
			searchQuery = searchQuery + "|" + keyword 

	results = api.search_submissions(q=searchQuery, before=endEpoch, after=startEpoch, limit=10)
	submissions = list(results)

	for submission in submissions:
		sentimentDataIdx = findDataIdx(submission.created_utc, epochData, startEpochIdx, endEpochIdx)

		print(sentimentDataIdx, end=", ")

		hasKeyword = False
		if sentimentDataIdx != -1:
			for keyword in keywords:
				caseKeyword = keyword.lower()
				if (caseKeyword in submission.title.lower()) or (caseKeyword in submission.selftext.lower()):
					hasKeyword = True
					break
		if hasKeyword:
#			print("-------- TITLE --------")
#			print(submission.title.encode())
#			print("-------- CONTENT --------")
#			print(submission.selftext.encode())
#			print("-------------------------")
			outMessageCount[sentimentDataIdx] += 1;

			sentimentScore = 0

			if submission.is_self:
				sentimentScore = GetSentiments.AnalyseSentiment(submission.title) * 0.5
				sentimentScore += GetSentiments.AnalyseSentiment(submission.selftext) * 0.5
			else:
				sentimentScore = GetSentiments.AnalyseSentiment(submission.title)

			if sentimentScore < 0:
				print(f"score = {sentimentScore}")
				print("-------- TITLE --------")
				print(submission.title.encode())
				print("-------- CONTENT --------")
				if submission.is_self:
					print(submission.selftext.encode())
				print("-------------------------")

			outSentimentScore[sentimentDataIdx] = sentimentScore
	print()
