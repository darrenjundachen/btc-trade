def percentage_seq_data(seq_data):
    """
    Data to be the percentage change of previous one
    First line will be removed as result
    """
    result = []
    for index in range(1, len(seq_data)):
        result.append([])
        for data_index in range(len(seq_data[index])):
            if seq_data[index - 1][data_index] == 0:
                result[index - 1].append(0)
            else:
                result[index - 1].append(
                    (seq_data[index][data_index] - seq_data[index - 1][data_index])
                    / seq_data[index - 1][data_index]
                )
    return result