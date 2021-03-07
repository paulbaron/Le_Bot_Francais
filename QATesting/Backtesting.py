from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
from os import listdir, path
from os.path import isfile, join
from backtesting.test import SMA, GOOG

DATAFOLDER = f'NoUploadData'

def loadDataFrame(name):
    df = pd.read_feather(name)
    print(df)
    df.set_index("timestamp", inplace=True)
    return df

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()

filesNames = [f for f in listdir(DATAFOLDER) if isfile(join(DATAFOLDER, f))]
filesNames = [f for f in filesNames if '^' not in f]

for i in range(len(filesNames)):
    filesNames[i] = filesNames[i].replace('.fed', '')

loaded_df = loadDataFrame(f'{DATAFOLDER}/{filesNames[55]}.fed')
loaded_df = loaded_df.rename(columns={"open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume"})
loaded_df = loaded_df.sort_index()[:1000] # hard limite for the first 5000 minutes
print(loaded_df)

bt = Backtest(loaded_df, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()