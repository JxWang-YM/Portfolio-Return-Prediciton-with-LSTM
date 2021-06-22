import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def train_valid_test_split(data, train_size=0.7, valid_size=0.15):
    data = data.dropna(axis=0)

    n = len(data)
    train_idx = round(train_size * n)
    valid_idx = round(valid_size * n)
    test_idx = n - train_idx - valid_idx

    train = data[:train_idx]
    valid = data[train_idx - 12:train_idx + valid_idx]
    test = data[-test_idx - 12:]
    return train, valid, test


def normalization(train, test=None, normalize_by_train=False,
                  seg_num=1,
                  EMA_smoothing=False, EMA_window=10, smoothing_param=2):

    scaler = MinMaxScaler(feature_range=(0, 1))
    arr_train_normalized = np.zeros_like(train).reshape(-1, 1)
    n = len(train)
    normalizing_window = round(n/seg_num)

    # due to half-open nature of range(), need to add n_w to t_s to access last seg of train data
    for j in range(0, n+normalizing_window, normalizing_window):
        if j == 0:
            i = j
            continue

        seg = train[i:j].to_numpy().reshape(-1, 1)
        arr_seg_normalized = scaler.fit_transform(seg)
        arr_train_normalized[i:j] = arr_seg_normalized
        i = j

    if normalize_by_train:
        test = test.to_numpy().reshape(-1, 1)
        arr_test_normalized = scaler.transform(test)

        return arr_test_normalized

    if EMA_smoothing:
        arr_smoothed = np.zeros_like(arr_train_normalized)
        EMA_multiplier = smoothing_param / (1 + EMA_window)

        EMA = 0

        for t in range(arr_train_normalized.shape[0]):
            EMA = arr_train_normalized[t] * EMA_multiplier + EMA * (1 - EMA_multiplier)
            arr_smoothed[t] = EMA

        return arr_smoothed

    return arr_train_normalized


def X_y_by_stock(arr, X_seq_length=12, y_forward=1):
    row_size = arr.shape[0] - X_seq_length

    X = np.zeros((row_size, X_seq_length))
    y = np.zeros((row_size, 1))
    for i in range(row_size):
        X[i, :] = arr[i:i + X_seq_length].reshape(1, X_seq_length)
        y[i, :] = arr[i + X_seq_length + y_forward - 1]

        torch_X = torch.from_numpy(X).float()
        torch_y = torch.from_numpy(y).float()

    return torch_X, torch_y


def save_numpy(*args):
    for arr in args:
        arr_name = [k for k, v in globals().items() if v is arr]  # if locals() is used here, arr_name would be arr
        np.save('{}.npy'.format(arr_name[0]), arr)


stock_price = pd.read_csv('stock_price.csv', index_col='Date')

df_train_dic = {}

training_dic = {}
validation_dic = {}
test_dic = {}

tickers = stock_price.columns

X_seq_length = 12
y_forward = 1

for ticker in tickers:
    train, valid, test = train_valid_test_split(stock_price[ticker])

    df_train_dic[ticker] = train.to_numpy()

    arr_train_normalized = normalization(train)
    arr_valid_normalized = normalization(train, valid, normalize_by_train=True)
    arr_test_normalized = normalization(train, test, normalize_by_train=True)

    training_dic[ticker] = X_y_by_stock(arr_train_normalized, X_seq_length, y_forward)
    validation_dic[ticker] = X_y_by_stock(arr_valid_normalized, X_seq_length, y_forward)
    test_dic[ticker] = X_y_by_stock(arr_test_normalized, X_seq_length, y_forward)

df_train = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df_train_dic.items()]))

df_train.to_csv('df_train.csv')

save_numpy(training_dic)
save_numpy(validation_dic)
save_numpy(test_dic)










