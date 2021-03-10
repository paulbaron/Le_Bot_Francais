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
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
EPOCHS = 5  # how many passes through our data
RATIOS_LEN = 15
BATCH_SIZE =  2048  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"IND_DATA-LEN-{RATIOS_LEN}-SEQ-{SEQ_LEN}"

filesNames = [f for f in listdir(f"{DATAFOLDER}/__WithIndicators") if isfile(join(f"{DATAFOLDER}/__WithIndicators", f))]
filesNames = [f for f in filesNames if '^' not in f]

for i in range(len(filesNames)):
    filesNames[i] = filesNames[i].replace('_WI.fed', '')

random.shuffle(filesNames)
RATIOS = filesNames[:RATIOS_LEN]

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


def preprocess_df(df):

    df = df.drop("future", 1)  # don't need this anymore.
    df = df.drop("close", 1)  # don't need this anymore.
    pd.set_option('use_inf_as_na', True)

    df.dropna(inplace=True)  # remove the nas created by pct_change
    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            if (col == "close"):
                df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
                pd.set_option('use_inf_as_na', True)
                df.dropna(inplace=True)  # remove the nas created by pct_change

    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!

global_train_x = []
global_train_y = []
global_validation_x = []
global_validation_y = []

for RATIO in RATIOS:
    print(f"Collecting ratio: {RATIO}")
    main_df = pd.DataFrame() # begin empty
    
    df = loadDataFrame(f'{DATAFOLDER}/__WithIndicators/{RATIO}_WI.fed')

    df = df[1:]
    #for col in df.columns:
    #    print(col)
    df = df[[f"close", f"MACD_MACD", f"MACD_SIGNAL", f"STOCHRSI", f"EBBP_Bull.", f"EBBP_Bear.", f"BASP_Buy.", f"BASP_Sell."]]  # ignore the other columns besides price and volume
    main_df = df  # then it's just the current df
    
    main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
    main_df.dropna(inplace=True)
    
    
    main_df['future'] = main_df[f'close'].shift(-FUTURE_PERIOD_PREDICT)
    main_df['target'] = list(map(classify, main_df[f'close'], main_df['future']))
    
    main_df.dropna(inplace=True)
    
    
    ## here, split away some slice of the future data from the main main_df.
    #times = sorted(main_df.index.values)
    #last_50pct = sorted(main_df.index.values)[-int(0.5*len(times))]
    #
    #main_df = main_df[(main_df.index < last_50pct)]
    
    times = sorted(main_df.index.values)
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

tensorboard = TensorBoard(log_dir="ModelTraining\logs\{}".format(NAME))

filepath = NAME + "-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("ModelTraining\models\{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones

# Train model
history = model.fit(
    global_train_x, global_train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(global_validation_x, global_validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Save model
model.save("ModelTraining\models\{}".format(NAME))

#tensorboard --logdir="le folder logs"
#tensorboard http://localhost:6006/