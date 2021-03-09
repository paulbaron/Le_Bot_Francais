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

DATAFOLDER = f'NoUploadData'
RATIO_TO_PREDICT = "IBM"
SEQ_LEN = 60 
NAME = f"DATA-LEN-10-SEQ-60"

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
        sub_df = self.data.df[[f"Close", f"High", f"Low", f"Volume"]]
        #print(sub_df)
        for col in sub_df.columns:
            sub_df[col] = sub_df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            pd.set_option('use_inf_as_na', True)
            sub_df.dropna(inplace=True)
        if (len(sub_df.values) == 0):
            return
        print(sub_df)
        input_data = np.array([n for n in sub_df.values[-1]])

        sequential_data = []
        prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

        if (len(sub_df.values) < SEQ_LEN):
            return
        last_60 = sub_df.values[-SEQ_LEN:]
        for i in last_60:  # iterate over the values
            prev_days.append([n for n in i])  # store all but the target
        sequential_data.append([prev_days])  # append those bad boys!


        sequential_data = np.asarray(sequential_data[0])
        #print(sequential_data.shape)
        #print(sequential_data)
        decision = self.model.predict(sequential_data)
        print(f"decision: {decision}")
        if decision[0][0] > 0.5 and self.lastWasBuy == False:
            self.buy()
            self.lastWasBuy = True
        elif decision[0][1] > 0.5 and self.lastWasBuy == True:
            self.sell()
            self.lastWasBuy = False

loaded_df = loadDataFrame(f'{DATAFOLDER}/{RATIO_TO_PREDICT}.fed')
loaded_df = loaded_df.rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
print(len(loaded_df))
loaded_df = loaded_df.sort_index()[200:800] # hard limite for the first 5000 minutes
print(loaded_df)

bt = Backtest(loaded_df, RNNStrategy, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()

