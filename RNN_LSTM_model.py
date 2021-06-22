import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class StockPricePredictor(nn.Module):

    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.input_size = input_size  # num of data labels within each element of a batch element
        self.hidden_layer_size = hidden_layer_size  # num of neurons in each hidden layer
        self.output_size = output_size
        self.num_layers = num_layers  # num of hidden layer

        self.lstm = nn.LSTM(
            # think of [input_size, hidden_size] as a weight matrix
            input_size=self.input_size, hidden_size=self.hidden_layer_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(
            # think of [in, out] as a weight matrix
            in_features=self.hidden_layer_size, out_features=self.output_size
        )

    def forward(self, input_seq):
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()

        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).requires_grad_()

        # [batch_size, seq_length, input_size]
        lstm_out, (hn, cn) = self.lstm(input_seq.view(input_seq.size(0), input_seq.size(1), self.input_size))

        prediction = self.linear(lstm_out[:, -1, :].view(input_seq.size(0), -1))  # .view(batch_size, -1)
        return prediction


def model_training(model, training_generator):
    loss_fn = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 80

    train_record = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        for local_X_train, local_y_train in training_generator:

            # similar with optimiser.zero_grad(), but acts on all parameters instead of
            # just the params in one paticular optimiser, which is what optimiser.zero_grad() does
            model.zero_grad()

            local_y_pred = model(local_X_train)
            loss = loss_fn(local_y_pred, local_y_train)

            loss.backward()
            optimiser.step()

        train_record[epoch] = loss.item()
        if epoch % 10 == 0 and epoch != 0:
            print('Epoch:', epoch, 'training MSE:', round(loss.item(), 5))
        elif epoch == num_epochs-1:
            print(' ')

    return train_record


def model_evaluation(model, validation_generator, scaler, use_pred=False, fut_pred=12):
    model.eval()

    predicted = np.array([]).reshape(-1, 1)
    actual = np.array([]).reshape(-1, 1)

    with torch.no_grad():
        for indx, data in enumerate(validation_generator, 0):
            local_X_valid, local_y_valid = data

            if use_pred:
                if indx == 0:
                    input_seq = local_X_valid

                local_y_pred = model(input_seq[-12:])
                input_seq = torch.cat((input_seq, local_y_pred), 1)

                predicted = np.append(predicted, local_y_pred, 0)
                actual = np.append(actual, local_y_valid, 0)

                if indx == fut_pred:
                    predicted = predicted[:fut_pred]
                    actual = actual[:fut_pred]
                    break

            else:
                local_y_pred = model(local_X_valid)

                predicted = np.append(predicted, local_y_pred, 0)
                actual = np.append(actual, local_y_valid, 0)

    unscaled_predicted = scaler.inverse_transform(predicted)
    unscaled_actual = scaler.inverse_transform(actual)

    return unscaled_predicted, unscaled_actual


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]


df_train = pd.read_csv('df_train.csv', index_col=0)

training_dic = np.load('training_dic.npy', allow_pickle=True).item()
validation_dic = np.load('validation_dic.npy', allow_pickle=True).item()
test_dic = np.load('test_dic.npy', allow_pickle=True).item()

model_dic = {}
scaler_dic = {}
train_record_dic = {}

tickers = list(training_dic.keys())
ticker_counts = len(tickers)

train_params = {'batch_size': 30,
                'shuffle': False,
                'num_workers': 6}

for ticker in tickers:
    model_dic[ticker] = StockPricePredictor()

    training_set = Dataset(training_dic[ticker])
    training_generator = torch.utils.data.DataLoader(training_set, **train_params)

    train_record_dic[ticker] = model_training(model_dic[ticker], training_generator)


# plot training MSE
fig, ax = plt.subplots(ticker_counts, figsize=(15, 13))
for t in range(ticker_counts):
    ax[t].plot(train_record_dic[tickers[t]])
    ax[t].set_title(tickers[t])
    ax[t].spines['right'].set_visible(False)
    ax[t].spines['top'].set_visible(False)

plt.suptitle('Training MSE by Epoch')
plt.xlabel('Epoch')
plt.show()


# Validation
valid_params = {'batch_size': 40,
                'shuffle': False,
                'num_workers': 1}

valid_params_use_pred = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 1}

valid_pred, valid_actual = {}, {}
valid_pred_use_pred, valid_actual_use_pred = {}, {}

for ticker in tickers:
    t = tickers.index(ticker)

    scaler_dic[ticker] = MinMaxScaler().fit(df_train[ticker].to_numpy().reshape(-1, 1))

    validation_set = Dataset(validation_dic[ticker])

    validation_generator = torch.utils.data.DataLoader(validation_set, **valid_params)
    validation_generator_use_pred = torch.utils.data.DataLoader(validation_set, **valid_params_use_pred)

    model = model_dic[ticker]

    valid_pred[ticker], valid_actual[ticker] = model_evaluation(model, validation_generator, scaler_dic[ticker])
    valid_pred_use_pred[ticker], valid_actual_use_pred[ticker] = model_evaluation(model, validation_generator_use_pred,
                                                                                  scaler_dic[ticker], use_pred=True)

# plot validation result
fig, ax = plt.subplots(ticker_counts, 2, figsize=(25, 15))

for ticker in tickers:
    t = tickers.index(ticker)
    
    ax[t, 0].plot(valid_pred[ticker], label='Predicted')
    ax[t, 0].plot(valid_actual[ticker], label='Actual')
    ax[t, 0].legend()
    ax[t, 0].set_title(tickers[t])
    ax[t, 0].spines['right'].set_visible(False)
    ax[t, 0].spines['top'].set_visible(False)

    ax[t, 1].plot(valid_pred_use_pred[ticker], label='Predicted')
    ax[t, 1].plot(valid_actual_use_pred[ticker], label='Actual')
    ax[t, 1].legend()
    ax[t, 1].set_title(tickers[t])
    ax[t, 1].spines['right'].set_visible(False)
    ax[t, 1].spines['top'].set_visible(False)

plt.suptitle('Walk Forward Validation vs. Predicted Value Forward Validation')
plt.show()


# Test
test_params = {'batch_size': 40,
               'shuffle': False,
               'num_workers': 1}

test_params_use_pred = {'batch_size': 1,
                        'shuffle': False,
                        'num_workers': 1}

test_pred, test_actual = {}, {}
test_pred_use_pred, test_actual_use_pred = {}, {}

for ticker in tickers:
    t = tickers.index(ticker)

    scaler_dic[ticker] = MinMaxScaler().fit(df_train[ticker].to_numpy().reshape(-1, 1))

    test_set = Dataset(test_dic[ticker])

    test_generator = torch.utils.data.DataLoader(test_set, **test_params)
    test_generator_use_pred = torch.utils.data.DataLoader(test_set, **test_params_use_pred)

    model = model_dic[ticker]

    test_pred[ticker], test_actual[ticker] = model_evaluation(model, test_generator, scaler_dic[ticker])
    test_pred_use_pred[ticker], test_actual_use_pred[ticker] = model_evaluation(model, test_generator_use_pred,
                                                                                scaler_dic[ticker], use_pred=True)

# plot test result
fig, ax = plt.subplots(ticker_counts, 2, figsize=(25, 15))

for ticker in tickers:
    t = tickers.index(ticker)

    trmse = np.sqrt(train_record_dic[ticker][-1])
    vrmse = np.sqrt(mean_squared_error(valid_actual[ticker], valid_pred[ticker]))
    termse = np.sqrt(mean_squared_error(test_actual[ticker], test_pred[ticker]))

    print(
        '{t:>5}: Training-RMSE: {trmse:.3f}; Validation-RMSE: {vrmse:.3f}; Test-RMSE: {termse:.3f} \n'.format(t=ticker,
                                                                                                              trmse=trmse,
                                                                                                              vrmse=vrmse,
                                                                                                              termse=termse))
    ax[t, 0].plot(test_pred[ticker], label='Predicted')
    ax[t, 0].plot(test_actual[ticker], label='Actual')
    ax[t, 0].legend()
    ax[t, 0].set_title(tickers[t])
    ax[t, 0].spines['right'].set_visible(False)
    ax[t, 0].spines['top'].set_visible(False)

    ax[t, 1].plot(test_pred_use_pred[ticker], label='Predicted')
    ax[t, 1].plot(test_actual_use_pred[ticker], label='Actual')
    ax[t, 1].legend()
    ax[t, 1].set_title(tickers[t])
    ax[t, 1].spines['right'].set_visible(False)
    ax[t, 1].spines['top'].set_visible(False)

plt.suptitle('Walk Forward Test vs. Predicted Value Forward Test')
plt.show()

# export stock return
stock_price_pred = pd.DataFrame(dict([(k, pd.Series(v.flatten())) for k, v in test_pred.items()]))
# stock_price_pred = stock_return_pred.dropna(axis=0)
stock_return_pred = stock_price_pred.pct_change()
stock_return_pred.to_csv('stock_return_pred.csv')