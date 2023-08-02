import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from scipy.optimize import minimize

# Set the analysis period
start_date = '2010-01-01'
end_date = '2021-12-31'

# Download historical ETF data
tickers = ['VT', 'SPY', 'VGK', 'EWU', 'MCHI', 'EEM', 'EWJ', 'AAXJ', 'QQQ', 'MTUM', 'QUAL', 'VLUE', 'DBC', 'TLT']
data = pd.DataFrame(columns=tickers)

for t in tickers:
    try:
        data[t] = pdr.DataReader(t, 'yahoo', start_date, end_date)['Adj Close']
    except:
        data[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate expected returns and covariances of returns
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Define the number of random wallets to generate
num_portfolios = 10000

# Define the function to calculate the variance of the portfolio
def portfolio_volatility(weights, cov_matrix):
    """
    Calculate the variance of the portfolio given the asset weights and the covariance matrix.
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)

# Generate random portfolios with random weights for ETFs
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    # Genera pesi casuali per gli ETF
    weights = np.array(np.random.random(len(tickers)))
    weights = weights / np.sum(weights)
    all_weights[i, :] = weights

    # Calculate the returns and variances of the generated portfolio
    ret_arr[i] = np.sum(mean_returns * weights) * 252
    vol_arr[i] = portfolio_volatility(weights, cov_matrix) * np.sqrt(252)

# Calculate portfolio with maximum Sharpe ratio
rf_rate = 0.01  # Tasso di interesse senza rischio
sharpe_arr = (ret_arr - rf_rate) / vol_arr
max_sharpe_idx = sharpe_arr.argmax()
optimal_weights = all_weights[max_sharpe_idx, :]
optimal_returns = ret_arr[max_sharpe_idx]
optimal_volatility = vol_arr[max_sharpe_idx]

# Calculate the efficient frontier
frontier_returns = []
frontier_volatility = []
frontier_portfolio_components = []
for r in np.linspace(0, 0.3, 100):
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - r})
    bnds = tuple((0, 1) for x in range(len(tickers)))
    res = minimize(portfolio_volatility, np.random.random(len(tickers)), args=(cov_matrix,), method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        continue
    weights = res.x
    frontier_returns.append(r)
    frontier_volatility.append(portfolio_volatility(weights, cov_matrix) * np.sqrt(252))
    frontier_portfolio_components.append(weights)

# Visualize the efficient frontier and the optimal portfolio
plt.figure(figsize=(12, 8))
plt.scatter(vol_arr*100, ret_arr*100, alpha=0.2, label='Portafogli casuali')
plt.scatter(optimal_volatility*100, optimal_returns*100, color='red', marker='*', s=300, label='Portafoglio ottimale')
plt.plot(frontier_volatility*100, frontier_returns*100, color='blue', linestyle='--', linewidth=2, label='Frontiera efficiente')
plt.title('Frontiera efficiente')
plt.xlabel('Volatilità (%)')
plt.ylabel('Rendimento atteso (%)')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.show()

# Create a table with portfolios and their returns and volatilities
portfolio_table = pd.DataFrame(columns=['Rendimento atteso', 'Volatilità', 'VT', 'SPY', 'VGK', 'EWU', 'MCHI', 'EEM', 'EWJ', 'AAXJ', 'QQQ', 'MTUM', 'QUAL', 'VLUE', 'DBC', 'TLT'])

for i in range(num_portfolios):
    portfolio_table.loc[i] = [ret_arr[i], vol_arr[i]] + list(all_weights[i, :])

# View the table
print(portfolio_table)

# Displays optimal portfolio weights
print('Pesi del portafoglio ottimale:')
for i in range(len(optimal_weights)):
    print(tickers[i], ':', round(optimal_weights[i]*100, 2), '%')
