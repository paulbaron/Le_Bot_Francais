import json

f = open("APIKeys.json")
apiKeys = json.load(f)

def GetAPIKey():
	return apiKeys