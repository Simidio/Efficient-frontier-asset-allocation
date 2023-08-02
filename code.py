import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from scipy.optimize import minimize

# Imposta il periodo di analisi
start_date = '2010-01-01'
end_date = '2021-12-31'

# Scarica i dati storici degli ETF
tickers = ['VT', 'SPY', 'VGK', 'EWU', 'MCHI', 'EEM', 'EWJ', 'AAXJ', 'QQQ', 'MTUM', 'QUAL', 'VLUE', 'DBC', 'TLT']
data = pd.DataFrame(columns=tickers)

for t in tickers:
    try:
        data[t] = pdr.DataReader(t, 'yahoo', start_date, end_date)['Adj Close']
    except:
        data[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']

# Calcola i rendimenti giornalieri
returns = data.pct_change().dropna()

# Calcola i rendimenti attesi e le covarianze dei rendimenti
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Definisci il numero di portafogli casuali da generare
num_portfolios = 10000

# Definisci la funzione per calcolare la varianza del portafoglio
def portfolio_volatility(weights, cov_matrix):
    """
    Calcola la varianza del portafoglio dati i pesi degli asset e la matrice di covarianza.
    """
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)

# Genera portafogli casuali con pesi casuali per gli ETF
all_weights = np.zeros((num_portfolios, len(tickers)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)

for i in range(num_portfolios):
    # Genera pesi casuali per gli ETF
    weights = np.array(np.random.random(len(tickers)))
    weights = weights / np.sum(weights)
    all_weights[i, :] = weights

    # Calcola i rendimenti e le varianze del portafoglio generato
    ret_arr[i] = np.sum(mean_returns * weights) * 252
    vol_arr[i] = portfolio_volatility(weights, cov_matrix) * np.sqrt(252)

# Calcola il portafoglio con il massimo rapporto di Sharpe
rf_rate = 0.01  # Tasso di interesse senza rischio
sharpe_arr = (ret_arr - rf_rate) / vol_arr
max_sharpe_idx = sharpe_arr.argmax()
optimal_weights = all_weights[max_sharpe_idx, :]
optimal_returns = ret_arr[max_sharpe_idx]
optimal_volatility = vol_arr[max_sharpe_idx]

# Calcola la frontiera efficiente
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

# Visualizza la frontiera efficiente e il portafoglio ottimale
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

# Crea una tabella con i portafogli e i loro rendimenti e volatilità
portfolio_table = pd.DataFrame(columns=['Rendimento atteso', 'Volatilità', 'VT', 'SPY', 'VGK', 'EWU', 'MCHI', 'EEM', 'EWJ', 'AAXJ', 'QQQ', 'MTUM', 'QUAL', 'VLUE', 'DBC', 'TLT'])

for i in range(num_portfolios):
    portfolio_table.loc[i] = [ret_arr[i], vol_arr[i]] + list(all_weights[i, :])

# Visualizza la tabella
print(portfolio_table)

# Visualizza i pesi del portafoglio ottimale
print('Pesi del portafoglio ottimale:')
for i in range(len(optimal_weights)):
    print(tickers[i], ':', round(optimal_weights[i]*100, 2), '%')
