from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def AnalyseSentiment(text):
	kvp = sia.polarity_scores(text)
	return kvp["compound"] 
