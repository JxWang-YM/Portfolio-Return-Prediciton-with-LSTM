import sys
import quandl
import pandas as pd
import matplotlib.pyplot as plt

quandl.read_key(filename='.quandl_apikey')

tickers = ['AAL', 'DIS', 'F', 'MSFT']
data = pd.DataFrame()

# From WIKI Dataset, discontinued, experiment only
# data = quandl.get_table('WIKI/PRICES', ticker=tickers,
#                         qopts={'columns': ['date', 'ticker', 'adj_close']},
#                         date={'gte': '2010-01-01', 'lte': '2018-03-27'},
#                         paginate=True)

# From EOD Dataset, subscription needed
for ticker in tickers:
    ind_stock = quandl.get('EOD/{}'.format(ticker), start_date='2000-01-01', end_date='2020-07-01',
                           collapse='monthly',
                           column_index=11)
    ind_stock['Ticker'] = ticker
    data = data.append(ind_stock)

data = data.pivot_table('Adj_Close', index='Date', columns='Ticker')


fig, ax = plt.subplots(figsize=(12, 8))

for c in data.columns:
    ax.plot(data.index, data[c], label=c)

ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend(loc='upper left')

plt.show()

data.to_csv('{}/stock_price.csv'.format(sys.argv[1]))












