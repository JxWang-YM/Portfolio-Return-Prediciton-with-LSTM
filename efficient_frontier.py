import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def random_portfolio_characters(avg_return, cov_matrix, risk_free_rate):

    def random_weights_gen():
        global port_sample_size

        num_stock = len(stock_return.columns)
        weights = np.random.rand(port_sample_size, num_stock)
        weights /= weights.sum(axis=1).reshape(port_sample_size, -1)
        return weights

    weights = random_weights_gen()

    annual_return = np.dot(weights, avg_return) * 12
    annual_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)).diagonal()) * np.sqrt(12)
    sharp_ratio = (annual_return - risk_free_rate) / annual_risk

    rand_port_char = pd.DataFrame(np.concatenate((annual_return.reshape(-1, 1), annual_risk.reshape(-1, 1), sharp_ratio.reshape(-1, 1)), axis=1),
                                  columns=['annual_return', 'annual_risk', 'sharp_ratio'])
    rand_port_char.sort_values('annual_risk', inplace=True)
    rand_port_char = rand_port_char.round(decimals=3)
    return rand_port_char


def port_optimization(avg_return, cov_matrix, target_return=None, obj='min_var'):

    def port_return(weight):
        nonlocal avg_return
        annual_return = np.dot(weight, avg_return) * 12
        return annual_return

    def port_risk(weight):
        nonlocal cov_matrix
        annual_risk = np.sqrt(np.dot(weight, np.dot(cov_matrix, weight.T))) * np.sqrt(12)
        return annual_risk

    def neg_sharp_ratio(weight):
        global risk_free_rate
        sr = -(port_return(weight) - risk_free_rate) / port_risk(weight)
        return sr

    if obj == 'min_var':
        obj_fun = port_risk
    if obj == 'max_sr':
        obj_fun = neg_sharp_ratio

    num_stock = len(avg_return)
    bounds = tuple((0, 1) for stock in range(num_stock))

    if target_return:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # sum of weight equals to 1
                       {'type': 'eq', 'fun': lambda x: port_return(x) - target_return})  # port_return equals to target_return
    else:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum of weight equals to 1

    result = minimize(obj_fun, num_stock*[1/num_stock], bounds=bounds, constraints=constraints, method='SLSQP')
    return result


def efficient_frontier(avg_return, cov_matrix):
    n = 50
    r_min = np.dot(port_optimization(avg_return, cov_matrix)['x'], avg_return) * 12
    r_max = avg_return.max() * 12

    return_range = np.linspace(r_min, r_max, n)
    risk_range = np.empty((n, 1))

    for i in range(n):
        result = port_optimization(avg_return, cov_matrix, return_range[i])
        opt_risk = result['fun']
        risk_range[i] = opt_risk

    df_ef = pd.DataFrame(np.concatenate((return_range.reshape(-1, 1), risk_range), axis=1),
                         columns=['ef_return', 'ef_risk'])
    return df_ef


def max_sr(avg_return, cov_matrix):
    max_sr_allocation = port_optimization(avg_return, cov_matrix, obj='max_sr')['x']
    max_sr_risk = np.sqrt(np.dot(max_sr_allocation, np.dot(cov_matrix, max_sr_allocation.T))) * np.sqrt(12)
    max_sr_return = np.dot(max_sr_allocation, avg_return) * 12
    return max_sr_allocation, max_sr_risk, max_sr_return


def min_var(avg_return, cov_matrix):
    min_var_allocation = port_optimization(avg_return, cov_matrix)['x']
    min_var_risk = port_optimization(avg_return, cov_matrix)['fun']
    min_var_return = np.dot(min_var_allocation, avg_return) * 12
    return min_var_allocation, min_var_risk, min_var_return


def efficient_frontier_graph(avg_return, cov_matrix, rand_port_char):
    df_ef = efficient_frontier(avg_return, cov_matrix)

    max_sr_allocation, max_sr_risk, max_sr_return = max_sr(avg_return, cov_matrix)
    min_var_allocation, min_var_risk, min_var_return = min_var(avg_return, cov_matrix)

    annual_risk_ind = np.sqrt(cov_matrix.to_numpy().diagonal()) * np.sqrt(12)
    annual_return_ind = avg_return * 12

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 7))

    # random portfolios
    sca = ax.scatter(rand_port_char.annual_risk, rand_port_char.annual_return,
                     s=10, alpha=.4, c=rand_port_char.sharp_ratio, cmap='PuBu')

    # efficient frontier
    ax.plot(df_ef['ef_risk'], df_ef['ef_return'],
            linestyle='--', color='grey', linewidth=3, label='Efficient Frontier')

    # individual assets risk-return profile
    ax.scatter(annual_risk_ind, annual_return_ind, s=50, color='crimson')
    for i, text in enumerate(avg_return.index):
        ax.annotate(text, (annual_risk_ind[i], annual_return_ind[i]),
                    xytext=(0, -18), textcoords='offset points', ha='center', va='bottom')

    # maximum sharp ratio point
    ax.scatter(max_sr_risk, max_sr_return,
               color='orangered', s=70, marker='X', label='Maximum Sharp Ratio', zorder=5)

    # minimum variance point
    ax.scatter(min_var_risk, min_var_return,
               color='purple', s=70, marker='X', label='Minimum Risk', zorder=5)

    # color bar set up
    clb = fig.colorbar(sca)
    clb.ax.set_title('Sharp Ratio', fontsize=10)

    plt.xlabel('Annualized Risk')
    plt.ylabel('Annualized Return')
    plt.legend(loc='upper left')
    plt.suptitle('Portfolio Optimization with Efficient Frontier', ha='center', fontsize=18)
    plt.title('Assets: ' + ', '.join(avg_return.index.to_list()), fontsize=10)

    plt.show()


def efficient_frontier_report(avg_return, cov_matrix):
    global risk_free_rate

    max_sr_allocation, max_sr_risk, max_sr_return = max_sr(avg_return, cov_matrix)
    min_var_allocation, min_var_risk, min_var_return = min_var(avg_return, cov_matrix)

    df_max_sr_allocation = pd.DataFrame(np.vectorize(round)(max_sr_allocation.reshape(1, -1), 2),
                                        columns=cov_matrix.columns, index=['Allocation'])
    df_min_var_allocation = pd.DataFrame(np.vectorize(round)(min_var_allocation.reshape(1, -1), 2),
                                         columns=cov_matrix.columns, index=['Allocation'])

    annual_risk_ind = np.vectorize(round)(np.sqrt(cov_matrix.to_numpy().diagonal()) * np.sqrt(12), 2)
    annual_return_ind = np.vectorize(round)(avg_return * 12, 2)
    df_ind_stock = pd.DataFrame(np.concatenate((annual_return_ind.reshape(1, -1), annual_risk_ind.reshape(1, -1)), axis=0),
                                columns=cov_matrix.columns, index=['Annualized Return', 'Annualized Volatility'])

    n = 50
    print('-'*n)
    print('Portfolio with Maximum Sharp Ratio')
    print('\n')
    print('Risk Free Rate: {input: .{digit}f}'.format(input=risk_free_rate, digit=2))
    print('\n')
    print('Annualized Return: {input:.{digit}f}'.format(input=max_sr_return, digit=2))
    print('Annualized Volatility: {input:.{digit}f}'.format(input=max_sr_risk, digit=2))
    print('\n')
    print(df_max_sr_allocation)
    print('-'*n)
    print('Portfolio with Minimum Volatility')
    print('\n')
    print('Annualized Return: {input:.{digit}f}'.format(input=min_var_return, digit=2))
    print('Annualized Volatility: {input:.{digit}f}'.format(input=min_var_risk, digit=2))
    print('\n')
    print(df_min_var_allocation)
    print('-'*n)
    print('Individual Stock Profile')
    print('\n')
    print(df_ind_stock)
    print('-'*n)


port_sample_size = 10000
risk_free_rate = .02

stock_return = pd.read_csv('stock_return_pred.csv', index_col=0)
avg_return = stock_return.mean(axis=0)
cov_matrix = stock_return.cov()

rand_port_char = random_portfolio_characters(avg_return, cov_matrix, risk_free_rate)

efficient_frontier_report(avg_return, cov_matrix)
efficient_frontier_graph(avg_return, cov_matrix, rand_port_char)



