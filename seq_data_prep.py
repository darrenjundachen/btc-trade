from constants import SEQ_LEN, TARGET_MINS

seq_data = []
for index in range(len(df) - 1, int(1440 / 5) * (SEQ_LEN - 1), -1):
    prev_data = []
    for seq_index in range(SEQ_LEN):
        pre_row = []
        for steps in (5, 60, 1440):
            pre_row.append(df[f"low_{steps}"][index - seq_index * int(steps / 5)])
            pre_row.append(df[f"high_{steps}"][index - seq_index * int(steps / 5)])
            pre_row.append(df[f"open_{steps}"][index - seq_index * int(steps / 5)])
            pre_row.append(df[f"close_{steps}"][index - seq_index * int(steps / 5)])
            pre_row.append(
                df[f"volume_{steps}"][index - seq_index * int(steps / 5)]
            )
        prev_data.append(pre_row)
    prev_data.reverse()

        

    seq_data.append([prev_data, df[f"target_{TARGET_MINS}"][index]])