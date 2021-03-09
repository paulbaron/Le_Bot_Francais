# Access and read API contents
# Have to read 3 APIs at a time due to limitations of Alpha Vantage API for free use (max 5 accesses per minute)
# Unable to read all 6 stock APIs (FAANGM) at once

import sys
from io import StringIO
import urllib.request, json
from urllib.error import URLError, HTTPError
import time
import datetime
import pandas as pd
import GetAPIKey
import GetRandomNasdaqCompany
from os import path
# urllib is for opening URLs to get info
# json is for reading raw string from URL into json file
# time is for waiting 1 min before getting more API data
# Limit of 5 API calls per min, so need to run 3 by 3

DATAFOLDER = f'NoUploadData'

keys = GetAPIKey.GetAPIKey()

def dateTimeToTimestamp(dtIndex):
    return int(time.mktime(datetime.datetime.strptime(dtIndex,"%Y-%m-%d %H:%M:%S").timetuple()))

def stocks(str1='AAPL', year='1', month='1'):
    print("Stock for " + str1 + " year: " + year + " month: " + month)
    # save url inside variable as raw string
    url = r"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=" + str1
    url += r"&interval=5min"
    url += r"&slice=year" + year + r"month" + month
    url += r"&apikey=" + keys['AlphaVantage_Key']

    # use urllib.request.urlopen() to access API from URL links
    try:
        response = urllib.request.urlopen(url, timeout=20)
    except response.URLError as e:
        return False, DataFrame()
    # from var saved (HTTPresponse type), use .read() + .decode('utf-8')
    string = StringIO(response.read().decode('utf-8'))

    sub_df = pd.read_csv(string, sep=",")

    if len(sub_df.index) == 1:
        return True, sub_df
    if 'time' in sub_df:
        sub_df['timestamp'] = list(map(dateTimeToTimestamp, sub_df['time']))
        sub_df = sub_df.drop(columns=['time'])
        sub_df.set_index("timestamp", inplace=True)
    else:
        print("Print the note!!!!")
        print(sub_df[0])
    return True, sub_df

def loadDataFrame(name): #tmp.fed   
    df = pd.read_feather(name)
    df.set_index("timestamp", inplace=True)
    return df

def saveDataFrame(df, name):
    df = df.reset_index()
    df.to_feather(name)


symbols = GetRandomNasdaqCompany.GetNasdaqCompaniesSymbols(1)
symbols = symbols[0:120]

# remove already downloaded files
symbols = [f for f in symbols if path.exists(f'{DATAFOLDER}/{f}.fed') == False]

for symbol in symbols:
    df = pd.DataFrame()
    keepGather = True
    step = 1
    for year in range(1, 3, step):
        for month in range(1, 13, step):
            if keepGather:
                succeed, sub_df = stocks(symbol, f'{year}', f'{month}')
                if succeed:
                    step = 1 # yes we can continue
                    if len(sub_df.index) > 1:
                        df = df.append(sub_df)
                    else:
                        keepGather = False
                else:
                    step = 0 # retry this month
    
    print(df)

    print("SaveDataFrame")
    if len(df.index) > 1:
        saveDataFrame(df, f'{DATAFOLDER}/{symbol}.fed')
    #print("LoadDataFrame")
    #df = loadDataFrame(f'{DATAFOLDER}/{symbol}.fed')
    #print(df)