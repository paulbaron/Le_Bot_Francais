# Access and read API contents
# Have to read 3 APIs at a time due to limitations of Alpha Vantage API for free use (max 5 accesses per minute)
# Unable to read all 6 stock APIs (FAANGM) at once

import sys
from io import StringIO
import urllib.request, json
import time
import datetime
import pandas as pd
# urllib is for opening URLs to get info
# json is for reading raw string from URL into json file
# time is for waiting 1 min before getting more API data
# Limit of 5 API calls per min, so need to run 3 by 3

def dateTimeToTimestamp(dtIndex):
    return int(time.mktime(datetime.datetime.strptime(dtIndex,"%Y-%m-%d %H:%M:%S").timetuple()))

def stocks(str1='AAPL', year='1', month='1'):
    APIKEY_ALPHAVANTAGE = "RTGZGXQDPGAOQSHD"
    # save url inside variable as raw string
    url = r"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=" + str1
    url += r"&interval=5min"
    url += r"&slice=year" + year + r"month" + month
    url += r"&apikey=" + APIKEY_ALPHAVANTAGE

    # use urllib.request.urlopen() to access API from URL links
    response = urllib.request.urlopen(url)

    # from var saved (HTTPresponse type), use .read() + .decode('utf-8')
    string = StringIO(response.read().decode('utf-8'))
    df = pd.read_csv(string, sep=",")
    df['timestamp'] = list(map(dateTimeToTimestamp, df['time']))
    df = df.drop(columns=['time'])
    df.set_index("timestamp", inplace=True)
    print(df)
    return df

def loadDataFrame(name): #tmp.fed
    df = pd.read_feather('tmp.fed')
    df.set_index("timestamp", inplace=True)
    return df

def saveDataFrame(df, name='tmp.fed'):
    df = df.reset_index()
    df.to_feather('tmp.fed')


df = stocks('AAPL')
saveDataFrame(df, 'toto.fed')
df = loadDataFrame('toto.fed')

print(df)