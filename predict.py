import tensorflow as tf
import pandas as pd
from constants import SEQ_LEN
import numpy as np
import joblib
from binance_api import timestamp_to_datetime

df = pd.read_csv("data/klines.csv")

# Prep sequence data
last_row_index = len(df) - 1
seq_data = []
for seq_index in range(SEQ_LEN):
    trade_row = []
    for mins in (5, 60, 1440):
        end_index = last_row_index - mins * seq_index
        start_index = end_index - mins + 1
        trade_row.append(min(df["low"][start_index : end_index + 1]))
        trade_row.append(max(df["high"][start_index : end_index + 1]))
        trade_row.append(df["open"][start_index])
        trade_row.append(df["close"][end_index])
        trade_row.append(sum(df["volume"][start_index : end_index + 1]))
    seq_data.append(trade_row)
seq_data.reverse()

seq_data = np.array(seq_data)

model_dir = "training/2021_09_26_19_10/"

# Scale
scaler = joblib.load(f"{model_dir}scaler.gz")
seq_data = scaler.transform(seq_data)

# Predict
checkpoint_path = f"{model_dir}checkpoint"
model = tf.keras.models.load_model(checkpoint_path)
seq_data = np.reshape(seq_data, (-1, seq_data.shape[0], seq_data.shape[1]))

print(timestamp_to_datetime(df['close_time'][last_row_index]))
print(model.predict(seq_data))
