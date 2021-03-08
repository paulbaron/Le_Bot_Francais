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

# uncomment to force CPU
#tf.config.experimental.set_visible_devices([], "DML")

DATAFOLDER = f'NoUploadData'

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 15  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "IBM"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE =  2048  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{RATIO_TO_PREDICT}-SEQ-{SEQ_LEN}"

def loadDataFrame(name):
    df = pd.read_feather(name)
    print(df)
    df.set_index("timestamp", inplace=True)
    return df

def classify(current, future):
    is_greater = ((float(future) / (float(current))) > 1.003)
    if is_greater:  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df):
    #print(df)
    df = df.drop("future", 1)  # don't need this anymore.
    pd.set_option('use_inf_as_na', True)

    df.dropna(inplace=True)  # remove the nas created by pct_change
    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            #print(col)
            #df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            #pd.set_option('use_inf_as_na', True)
            #df.dropna(inplace=True)  # remove the nas created by pct_change
            #df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.
            #print(df[col])
            
            if col.endswith("_close"):
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)  # remove the nas created by pct_change
                max = df[col].values.max()
                print(f"old max: {max}")
                for i in range(len(df[col].values)):
                    df[col].values[i] = df[col].values[i] / max
            elif col.endswith("_high"):
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)  # remove the nas created by pct_change
                max = df[col].values.max()
                print(f"old max: {max}")
                for i in range(len(df[col].values)):
                    df[col].values[i] = df[col].values[i] / max
            elif col.endswith("_low"):
                df[col] = df[col].pct_change()
                df.dropna(inplace=True)  # remove the nas created by pct_change
                max = df[col].values.max()
                print(f"old max: {max}")
                for i in range(len(df[col].values)):
                    df[col].values[i] = df[col].values[i] / max
            else: # this is volume
                max = df[col].values.max()
                print(f"old max: {max}")
                for i in range(len(df[col].values)):
                    df[col].values[i] = df[col].values[i] / max
            #df.dropna(inplace=True)
            max = df[col].values.max()
            print(f"new max: {max}")

    #df.dropna(inplace=True)  # cleanup again... jic.

    print(df)

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


main_df = pd.DataFrame() # begin empty

df = loadDataFrame(f'{DATAFOLDER}/{RATIO_TO_PREDICT}.fed')

df.rename(columns={"close": f"{RATIO_TO_PREDICT}_close", "high": f"{RATIO_TO_PREDICT}_high", "low": f"{RATIO_TO_PREDICT}_low", "volume": f"{RATIO_TO_PREDICT}_volume"}, inplace=True)
df = df[1:]
df = df[[f"{RATIO_TO_PREDICT}_close", f"{RATIO_TO_PREDICT}_high", f"{RATIO_TO_PREDICT}_low", f"{RATIO_TO_PREDICT}_volume"]]  # ignore the other columns besides price and volume
main_df = df  # then it's just the current df

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)

print(main_df)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

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

print(main_df)
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

#print(validation_x)
#print(validation_y)
print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

print(train_x.shape[1:])
model = Sequential()
model.add(GRU(128, input_shape=(train_x.shape[1:]), return_sequences=True))
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

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

print(train_x)
# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("ModelTraining\models\{}".format(NAME))

#tensorboard --logdir="le folder logs"
#tensorboard http://localhost:6006/