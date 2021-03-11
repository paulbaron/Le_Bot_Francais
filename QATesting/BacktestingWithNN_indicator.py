from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from os import listdir, path
from os.path import isfile, join
from backtesting.test import SMA, GOOG
from collections import deque
from tensorflow import keras
import tensorflow as tf
import numpy as np
from finta import TA

DATAFOLDER = f'NoUploadData'
RATIO_TO_PREDICT = "JMIA"
RATIOS_LEN = 15
SEQ_LEN = 60 
NAME = f"IND_DATA-LEN-{RATIOS_LEN}-SEQ-{SEQ_LEN}"

# uncomment to force CPU
tf.config.experimental.set_visible_devices([], "DML")

def loadDataFrame(name):
    df = pd.read_feather(name)
    #print(df)
    df.set_index("timestamp", inplace=True)
    return df

class RNNStrategy(Strategy):

    def init(self):
        self.rolled_data = deque(maxlen=SEQ_LEN)
        model_name = f"ModelTraining\models\{NAME}"
        print("ModelTraining\models\{}".format(NAME))
        self.model = keras.models.load_model("ModelTraining\models\{}".format(NAME), compile=True)
        self.lastWasBuy = False

    def next(self):

        #print(self.data.df)

        #self.rolled_data.append([n for n in self.data.df.values[-1]])  # store all but the target
        pd.set_option('use_inf_as_na', True)
        indicators = ['MACD','STOCHRSI','EBBP','BASP']
        df = self.data.df#[[f"Close", f"High", f"Low", f"Volume"]]
        df = df.rename(columns={"Open":"open","Close":"close","High":"high","Low":"low","Volume":"volume"})
        #print(df)
        for ind in indicators:
            newdf = pd.DataFrame(getattr(TA, ind)(df))
            #print(newdf)
            if (len(newdf.columns) == 1):
                df[ind] = newdf
            else:
                for i in range(len(newdf.columns)):
                    newdf = newdf.rename(columns={newdf.columns[i]: f"{ind}_{newdf.columns[i]}"})
                df = pd.concat([df, newdf], axis=1)
        
        df = df[["MACD_MACD","MACD_SIGNAL","STOCHRSI","EBBP_Bull.","EBBP_Bear.","BASP_Buy.","BASP_Sell."]]
        df.dropna(inplace=True)  # remove the nas created by pct_change
        #print(df)
        if (len(df.values) == 0):
            return
        input_data = np.array([n for n in df.values[-1]])

        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

        if (len(df.values) < SEQ_LEN):
            return
        last_60 = df.values[-SEQ_LEN:]
        for i in last_60:  # iterate over the values
            prev_days.append([n for n in i])  # store all but the target
        sequential_data.append([prev_days])  # append those bad boys!


        sequential_data = np.asarray(sequential_data[0])
        #print(sequential_data.shape)
        #print(sequential_data)
        decision = self.model.predict(sequential_data)
        if decision[0][0] > 0.6 and self.lastWasBuy == False:
            print(f"decision: {decision} BUY")
            self.buy()
            self.lastWasBuy = True
        elif decision[0][1] > 0.6 and self.lastWasBuy == True:
            print(f"decision: {decision} SELL")
            self.sell()
            self.lastWasBuy = False
        else:
            print(f"decision: {decision}")

loaded_df = loadDataFrame(f'{DATAFOLDER}/{RATIO_TO_PREDICT}.fed')
loaded_df = loaded_df.rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
print(len(loaded_df))
loaded_df = loaded_df.sort_index()[800:2800] # hard limite for the first 5000 minutes
print(loaded_df)

bt = Backtest(loaded_df, RNNStrategy, commission=.002, # this need to be 0.002
              exclusive_orders=True)
stats = bt.run()
bt.plot()

