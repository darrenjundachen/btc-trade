import pandas as pd
import numpy as np
import random
import tensorflow as tf
from constants import SEQ_LEN, EPOCHS, BATCH_SIZE, LEARNING_RATE, TARGET_MINS
from datetime import datetime
from comm import percentage_seq_data


def prep_data(df, balance_data):
    # Add one more for percentage calculation
    extra_seq_len = SEQ_LEN + 1

    # Prep sequence data
    seq_data = []
    for index in range(len(df) - 1, int(1440 / 5) * (extra_seq_len - 1), -1):
        prev_data = []
        for seq_index in range(extra_seq_len):
            pre_row = []
            for steps in (5, 60, 1440):
                cell_index = index - seq_index * int(steps / 5)
                pre_row.append(df[f"low_{steps}"][cell_index])
                pre_row.append(df[f"high_{steps}"][cell_index])
                pre_row.append(df[f"open_{steps}"][cell_index])
                pre_row.append(df[f"close_{steps}"][cell_index])
                pre_row.append(df[f"volume_{steps}"][cell_index])
            prev_data.append(pre_row)
        prev_data.reverse()
        prev_data = percentage_seq_data(prev_data)
        seq_data.append([prev_data, df[f"target_{TARGET_MINS}"][index]])

        # Print progress
        if len(seq_data) % 10000 == 0:
            print(index)

    random.shuffle(seq_data)

    # Balance
    if balance_data:
        buys = []
        sells = []
        for seq, target in seq_data:
            if target == 0:
                sells.append([seq, target])
            elif target == 1:
                buys.append([seq, target])
        random.shuffle(buys)
        random.shuffle(sells)
        lower = min(len(buys), len(sells))
        buys = buys[:lower]
        sells = sells[:lower]
        seq_data = buys + sells

    random.shuffle(seq_data)

    # Return X and y
    X = []
    y = []
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


# Load data from csv
train_df = pd.read_csv("data/train_data.csv")
validation_df = pd.read_csv("data/validation_data.csv")

# Get timestamp for data folder
ts_name = datetime.now().strftime("%Y_%m_%d_%H_%M")
train_dir = f"training/{ts_name}/"

# Get train, validation data
train_x, train_y = prep_data(train_df, balance_data=False)
validation_x, validation_y = prep_data(validation_df, balance_data=False)
print(train_x[11])
# LSTM
model = tf.keras.Sequential(
    [
        tf.keras.layers.BatchNormalization(
            input_shape=(train_x.shape[1], train_x.shape[2])
        ),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=1e-6),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

checkpoint_path = f"{train_dir}checkpoint"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
)

model.fit(
    train_x,
    train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[cp_callback],
)
