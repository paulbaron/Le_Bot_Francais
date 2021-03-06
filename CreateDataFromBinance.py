from binance.client import Client
import pandas as pd
import numpy as np
import json 
from types import SimpleNamespace
import datetime as dt
import os.path
from os import path

client = Client("i5X7qL4NhbZafRbFyVJNjRpZVJJhd2f0y1tgSO1BN9AMCaxGUjL3PJHzVKRDei0k", "6qre5m0mgX0RXgSTtAjFl5EaTWs4KY2sJfToRF9Bv7FdB3kghCHHB7H6RsOCPbOx")

INTERVAL = Client.KLINE_INTERVAL_1HOUR

SYMBOL = 'BTCEUR'
FILEPATH = f"DATA/{SYMBOL}_{INTERVAL}.csv"
MAXROW = 1000000
df_len = 0

if path.exists(FILEPATH):
    fl = pd.read_csv(FILEPATH)
    oldestTimeInFile = fl['StartTime'][0]
    candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, endTime=oldestTimeInFile, limit=1000)
else:
    fl = pd.DataFrame(columns=["StartTime", "Open", "High", "Low", "Close", "Volume", "EndTime", "Quote", "NBTrades", "BaseAssetVolume", "QuoteAssetVolume", "Ignore"])
    candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=1000)

fl.set_index("StartTime", inplace=True) 
fl = fl[["Close", "High", "Low", "Volume"]]

while df_len < MAXROW:

    df = pd.DataFrame(data=candles, columns=["StartTime", "Open", "High", "Low", "Close", "Volume", "EndTime", "Quote", "NBTrades", "BaseAssetVolume", "QuoteAssetVolume", "Ignore"])
    oldestTimeInFile = df['StartTime'][0]
    df.set_index("StartTime", inplace=True) 
    df = df[["Close", "High", "Low", "Volume"]]

    df = df.append(fl)

    df_len = len(df)
    print(df_len)
    
    if (len(candles) < 1000): # we reach the end
        break

    #load prev 1000 rows
    candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, endTime=oldestTimeInFile, limit=1000)
    fl = df

if df_len < MAXROW:
    df.to_csv(FILEPATH)