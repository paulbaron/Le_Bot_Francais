import pandas as pd

nasdaqCompanies = pd.DataFrame()

fileData = pd.read_csv("../RawData/nasdaq_screener.csv", names=["Symbol", "Name", "Last Sale", "Net Change", "% Change", "Market Cap", "Country", "IPO Year", "Volume", "Sector", "Industry"])

fileData = fileData.drop(labels=["Last Sale", "Net Change", "% Change", "Country", "IPO Year", "Volume", "Industry"], axis=1)

fileData.to_csv("../RawData/AllNasdaqCompanies.csv", index=False)
