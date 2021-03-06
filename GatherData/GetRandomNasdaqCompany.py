import pandas as pd

def GetNasdaqCompaniesSymbols(seed):
	df = pd.DataFrame()
	df = pd.read_csv(filePath)
	df = df.sample(frac=1, random_state=seed)
	return df["Symbol"]