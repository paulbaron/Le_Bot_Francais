import pandas as pd

def GetNasdaqCompaniesSymbols(seed):
	df = pd.DataFrame()
	df = pd.read_csv("RawData/AllNasdaqCompanies.csv")
	df = df.sample(frac=1, random_state=seed)
	return df["Symbol"]