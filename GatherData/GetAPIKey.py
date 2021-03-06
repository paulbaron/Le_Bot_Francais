import json

f = open("GatherData/APIKeys.json")
apiKeys = json.load(f)

def GetAPIKey():
	return apiKeys