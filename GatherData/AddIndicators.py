import pandas as pd
import numpy as np
from finta import TA
from os import listdir, path
from os.path import isfile, join

DATAFOLDER = f'NoUploadData'

def loadDataFrame(name):
    df = pd.read_feather(name)
    df.set_index("timestamp", inplace=True)
    return df

def saveDataFrame(df, name):
    df = df.reset_index()
    df.to_feather(name)

filesNames = [f for f in listdir(DATAFOLDER) if isfile(join(DATAFOLDER, f))]
filesNames = [f for f in filesNames if '^' not in f]

for i in range(len(filesNames)):
    filesNames[i] = filesNames[i].replace('.fed', '')

# remove already computes files
filesNames = [f for f in filesNames if path.exists(f'{DATAFOLDER}/__WithIndicators/{f}_WI.fed') == False]

print(filesNames)

indicators = ['SMA','SMM','SSMA','EMA','DEMA','TEMA','TRIMA','TRIX','VAMA','ER','KAMA','ZLEMA','WMA','HMA','EVWMA','VWAP','SMMA','MACD','PPO','VW_MACD','EV_MACD','MOM','ROC','RSI','IFT_RSI','TR','ATR','BBANDS','BBWIDTH','MOBO','PERCENT_B','KC','DO','DMI','ADX','PIVOT','PIVOT_FIB','STOCH','STOCHD','STOCHRSI','WILLIAMS','UO','AO','MI','VORTEX','KST','TSI','TP','ADL','CHAIKIN','MFI','OBV','WOBV','VZO','PZO','EFI','CFI','EBBP','EMV','CCI','COPP','BASP','BASPN','CMO','CHANDELIER','QSTICK','WTO','FISH','ICHIMOKU','APZ','SQZMI','VPT','FVE','VFI','MSD','STC']
for fileName in filesNames:
    loaded_df = loadDataFrame(f'{DATAFOLDER}/{fileName}.fed')
    df = loaded_df
    for ind in indicators:
        #print(ind)
        newdf = pd.DataFrame(getattr(TA, ind)(loaded_df))
        if (len(newdf.columns) == 1):
            df[ind] = newdf
        else:
            for i in range(len(newdf.columns)):
                newdf = newdf.rename(columns={newdf.columns[i]: f"{ind}_{newdf.columns[i]}"})
            df = pd.concat([df, newdf], axis=1)

    print(df)
    saveDataFrame(df, f'{DATAFOLDER}/__WithIndicators/{fileName}_WI.fed')

#TA.FRAMA(df)
#TA.SAR(df)
#TA.TMF(df)




