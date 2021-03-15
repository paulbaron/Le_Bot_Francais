import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing
from os import listdir, path
from os.path import isfile, join

# uncomment to force CPU
#tf.config.experimental.set_visible_devices([], "DML")

DATAFOLDER = f'NoUploadData'

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 10  # how far into the future are we trying to predict?
EPOCHS = 5  # how many passes through our data
RATIOS_LEN = 5
BATCH_SIZE =  2048  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"RAW-LEN-{RATIOS_LEN}-SEQ-{SEQ_LEN}"

filesNames = [f for f in listdir(f"{DATAFOLDER}/__WithIndicators") if isfile(join(f"{DATAFOLDER}/__WithIndicators", f))]
filesNames = [f for f in filesNames if '^' not in f]

for i in range(len(filesNames)):
    filesNames[i] = filesNames[i].replace('_WI.fed', '')

# warning this is deterministic !!!!!!!
random.Random(10).shuffle(filesNames)
RATIOS = filesNames[:RATIOS_LEN]

# arrayLength => 101
AllindicatorColumns = ["SMA","SMM","SSMA","EMA","DEMA","TEMA","TRIMA","TRIX","VAMA","ER","KAMA","ZLEMA","WMA","HMA","EVWMA","VWAP","SMMA","MACD_MACD","MACD_SIGNAL","PPO_PPO","PPO_SIGNAL","PPO_HISTO","VW_MACD_MACD","VW_MACD_SIGNAL","EV_MACD_MACD","EV_MACD_SIGNAL","MOM","ROC","RSI","IFT_RSI","TR","ATR","BBANDS_BB_UPPER","BBANDS_BB_MIDDLE","BBANDS_BB_LOWER","BBWIDTH","MOBO_BB_UPPER","MOBO_BB_MIDDLE","MOBO_BB_LOWER","PERCENT_B","KC_KC_UPPER","KC_KC_LOWER","DO_LOWER","DO_MIDDLE","DO_UPPER","DMI_DI+","DMI_DI-","ADX","PIVOT_pivot","PIVOT_s1","PIVOT_s2","PIVOT_s3","PIVOT_s4","PIVOT_r1","PIVOT_r2","PIVOT_r3","PIVOT_r4","PIVOT_FIB_pivot","PIVOT_FIB_s1","PIVOT_FIB_s2","PIVOT_FIB_s3","PIVOT_FIB_s4","PIVOT_FIB_r1","PIVOT_FIB_r2","PIVOT_FIB_r3","PIVOT_FIB_r4","STOCH","STOCHD","STOCHRSI","WILLIAMS","UO","AO","MI","VORTEX_VIm","VORTEX_VIp","KST_KST","KST_signal","TSI_TSI","TSI_signal","TP","ADL","CHAIKIN","MFI","OBV","WOBV","VZO","PZO","EFI","CFI","EBBP_Bull.","EBBP_Bear.","EMV","CCI","COPP","BASP_Buy.","BASP_Sell.","BASPN_Buy.","BASPN_Sell.","CMO","CHANDELIER_Short.","CHANDELIER_Long."]
#AllindicatorColumns = AllindicatorColumns[:2]
#thoses are not good
#QSTICK
#WTO_WT1.
#WTO_WT2.
#FISH
#ICHIMOKU_TENKAN
#ICHIMOKU_KIJUN
#ICHIMOKU_senkou_span_a
#ICHIMOKU_SENKOU
#ICHIMOKU_CHIKOU
#APZ_UPPER
#APZ_LOWER
#SQZMI
#VPT
#FVE
#VFI
#MSD
#STC

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def loadDataFrame(name):
    df = pd.read_feather(name)
    df.set_index("timestamp", inplace=True)
    return df

def classify(current, future):
    is_greater = ((float(future) > (float(current))))
    if is_greater:  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0

def classifyWithRange(df):
    df_len = len(df.values)
    #print(df_len)
    df_target = [0.0] * df_len
    for i in range(df_len):
        sumGreater = 0
        sumLower = 0
        for j in range(1,min(FUTURE_PERIOD_PREDICT, df_len - i)):
            diff = df['close'][df.index[i + j]] - df['close'][df.index[i]]
            if diff > 0:
                sumGreater += pow(diff, 1)
            else:
                sumLower += abs(pow(diff, 1))

        if (sumGreater + sumLower) > 0:
            df_target[i] = float(sumGreater) / float(sumGreater + sumLower)
            #print(f"{sumGreater} {sumLower} {df_target[i]}")
        #if sumGreater > sumLower:
        #    df_target[i] = 1
        #else:
        #    df_target[i] = 0
#        print(df_target[i])
    df['target'] = df_target
    #print(df['target'])


def preprocess_df(df):

    df = df.drop("close", 1)  # don't need this anymore.
    pd.set_option('use_inf_as_na', True)

    df.dropna(inplace=True)  # remove the nas created by pct_change
    #for col in df.columns:  # go through all of the columns
    #    if col != "target":  # normalize all ... except for the target itself!
    #        #if (col == "close"):
    #        df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
    #df.dropna(inplace=True)  # remove the nas created by pct_change

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    #wait = []
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target < 0.20:  # if it's a "not buy"
            sells.append([seq, 0])  # append to sells list
        elif target > 0.80:  # otherwise if the target is a 1...
            buys.append([seq, 1])  # it's a buy!
        #else:
        #    wait.append([seq, 0.5])

    #print(f"buy {len(buys)} - wait {len(wait)} - sell {len(sells)}")

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells#+wait  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!
    return np.array(X), y  # return X and y...and make X a numpy array!

for indicatorID in range(len(AllindicatorColumns)):
    print(f"{OKGREEN}{UNDERLINE}TESTING => {AllindicatorColumns[indicatorID]} ({indicatorID + 1}/{len(AllindicatorColumns)}){ENDC}")
    NAME_WITH_INDICATOR = NAME + "__" + AllindicatorColumns[indicatorID]
    global_train_x = []
    global_train_y = []
    global_validation_x = []
    global_validation_y = []

    for RATIO in RATIOS:
        print(f"{OKCYAN}Collecting ratio: {RATIO}{ENDC}")

        main_df = pd.DataFrame() # begin empty

        df = loadDataFrame(f'{DATAFOLDER}/__WithIndicators/{RATIO}_WI.fed')
        df = df[1:]
        colnames = [f"close"] + [AllindicatorColumns[indicatorID]]
        #print(colnames)
        df = df[colnames]  # ignore the other columns besides price and volume
        main_df = df  # then it's just the current df

        main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
        main_df.dropna(inplace=True)


        classifyWithRange(main_df)
        #main_df['future'] = main_df[f'close'].shift(-FUTURE_PERIOD_PREDICT)
        #main_df['target'] = list(map(classify, main_df[f'close'], main_df['future']))

        main_df.dropna(inplace=True)

        times = sorted(main_df.index.values)
        if len(times) == 0:
            continue
        last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

        validation_main_df = main_df[(main_df.index >= last_5pct)]

        main_df = main_df[(main_df.index < last_5pct)]

        train_x, train_y = preprocess_df(main_df)
        validation_x, validation_y = preprocess_df(validation_main_df)

        if (len(train_x) == 0 or len(train_y) == 0 or  len(validation_x) == 0 or  len(validation_y) == 0):
            continue

        print(f"tx({len(train_x)}) ty({len(train_y)}) vx({len(validation_x)}) vy({len(validation_y)})")
        if (len(global_train_x) == 0):
            global_train_x = train_x
            global_train_y = train_y
            global_validation_x = validation_x
            global_validation_y = validation_y
        else:
            global_train_x = np.concatenate((global_train_x, train_x), axis=0)
            global_train_y = np.concatenate((global_train_y, train_y), axis=0)
            global_validation_x = np.concatenate((global_validation_x, validation_x), axis=0)
            global_validation_y = np.concatenate((global_validation_y, validation_y), axis=0)


    global_train_x = np.asarray(global_train_x)
    global_train_y = np.asarray(global_train_y)
    global_validation_x = np.asarray(global_validation_x)
    global_validation_y = np.asarray(global_validation_y)

    #print(validation_x)
    #print(validation_y)
    #print(global_train_y)
    print(f"train data: {len(global_train_x)} validation: {len(global_validation_x)}")
    #print(f"Dont buys: {global_train_y.count(0)}, buys: {global_train_y.count(1)}")
    #print(f"VALIDATION Dont buys: {global_validation_y.count(0)}, buys: {global_validation_y.count(1)}")



    print(global_train_x.shape[1:])
    model = Sequential()
    model.add(GRU(128, input_shape=(global_train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(GRU(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))


    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    tensorboard = TensorBoard(log_dir="ModelTraining\logs\_testIndicators\{}".format(NAME_WITH_INDICATOR))

    filepath = NAME_WITH_INDICATOR + "-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    #checkpoint = ModelCheckpoint("ModelTraining\models\_testIndicators\{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones

    # Train model
    history = model.fit(
        global_train_x, global_train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(global_validation_x, global_validation_y),
        callbacks=[tensorboard],
    )

    # Save model
    model.save("ModelTraining\models\_testIndicators\{}".format(NAME_WITH_INDICATOR))

    #tensorboard --logdir="le folder logs"
    #tensorboard http://localhost:6006/