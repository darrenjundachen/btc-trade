import pandas as pd
import numpy as np
from collections import defaultdict
from constants import TRAIN_TEST_SPLIT, PRICE_INC_RATE

def get_train_test_split():
    df = pd.read_csv("data/klines.csv")

    train_test_index = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df[:train_test_index]
    test_df = df[train_test_index:].reset_index()

    return train_df, test_df


def prep_trade_data(df, file_name):
    print(len(df) / 5)
    pd.DataFrame(
        columns=[
            "close_time",
            "low_5",
            "high_5",
            "open_5",
            "close_5",
            "volume_5",
            "low_60",
            "high_60",
            "open_60",
            "close_60",
            "volume_60",
            "low_1440",
            "high_1440",
            "open_1440",
            "close_1440",
            "volume_1440",
            "target_180",
            "target_360",
            "target_720",
            "target_1440",
        ]
    ).to_csv(file_name, index=False)

    # Prepare trade data
    starting_index = 1440 - 1  # starting index, 1 day
    trade_data = defaultdict(list)
    for index in range(starting_index, len(df) - 1440, 5):
        trade_data["close_time"].append(df["close_time"][index])

        # Get data prior to the close time for 5 mins, 1 hour, 1 day
        for mins in (5, 60, 1440):
            trade_data[f"low_{mins}"].append(
                min(df["low"][index - mins + 1 : index + 1])
            )
            trade_data[f"high_{mins}"].append(
                max(df["high"][index - mins + 1 : index + 1])
            )
            trade_data[f"open_{mins}"].append(df["open"][index - mins + 1])
            trade_data[f"close_{mins}"].append(df["close"][index])
            trade_data[f"volume_{mins}"].append(
                sum(df["volume"][index - mins + 1 : index + 1])
            )

        # Get target after close time for 3 hours, 6 hours, 12 hours, 24 hours
        for mins in (180, 360, 720, 1440):
            highest_price = max(df["high"][index + 1 : index + mins + 1])
            current_price = df["close"][index]
            trade_data[f"target_{mins}"].append(
                1 if highest_price >= current_price * (1 + PRICE_INC_RATE) else 0
            )

        if len(trade_data["close_time"]) % 10000 == 0:
            pd.DataFrame(trade_data).to_csv(
                file_name, mode="a", header=False, index=False
            )
            trade_data = defaultdict(list)
            print("added 10000")


train_df, test_df = get_train_test_split()

# Prep trade data
prep_trade_data(train_df, "data/train_data.csv")
prep_trade_data(test_df, "data/validation_data.csv")
