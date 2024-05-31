import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import openpyxl


# Lista de ativos
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'ABEV3.SA', 'MGLU3.SA',
           'BSBR', 'ERJ', 'BTC-USD', 'ETH-USD', 'XRP-USD']

# Período de 5 anos
start_date = '2019-01-01'
end_date = '2023-12-31'

# Coletar dados históricos de preços de fechamento ajustados
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calcular retornos logarítmicos
returns = np.log(data / data.shift(1))

# Dropar NaN resultantes do cálculo de retornos
returns = returns.dropna()

# Salvar os dados de preços históricos em um arquivo CSV
data.to_csv('historical_prices.csv')

# Salvar os retornos calculados em um arquivo CSV
returns.to_csv('returns.csv')

# Estatísticas descritivas dos retornos
stats = returns.describe().transpose()
stats['variance'] = returns.var()
stats['median'] = returns.median()

# Definir a taxa livre de risco (ex: Selic)
rf = 0.13 / 252  # Assumindo uma taxa Selic de 13% anual dividida por 252 dias úteis

# Calcular o Índice de Sharpe
stats['sharpe'] = (stats['mean'] - rf) / stats['std']

# Assumindo Ibovespa como o mercado
ibov = yf.download('^BVSP', start=start_date, end=end_date)['Adj Close']
ibov_returns = np.log(ibov / ibov.shift(1)).dropna()

# Alinhar as datas dos retornos dos ativos e do Ibovespa
aligned_returns = returns.reindex(ibov_returns.index).dropna()
aligned_ibov_returns = ibov_returns.reindex(aligned_returns.index)

# Calcular Beta
betas = {}
for ticker in tickers:
    cov_matrix = np.cov(aligned_returns[ticker], aligned_ibov_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    betas[ticker] = beta
stats['beta'] = pd.Series(betas)

# Calcular o Índice de Treynor
stats['treynor'] = (stats['mean'] - rf) / stats['beta']

# Calcular o VaR
confidence_level = 0.95
z = norm.ppf(confidence_level)
stats['var'] = stats['mean'] - z * stats['std']

# Exibir estatísticas descritivas
print(stats)

# Construir Portfólios

# Portfólio Ingênuo (alocação igual)
weights_ine = np.array([1/len(tickers)] * len(tickers))
returns_ine = aligned_returns.dot(weights_ine)

# Portfólio de Mínima Variância
cov_matrix = aligned_returns.cov()
inv_cov_matrix = np.linalg.inv(cov_matrix)
ones = np.ones(len(tickers))
weights_min_var = inv_cov_matrix.dot(ones) / ones.dot(inv_cov_matrix).dot(ones)
returns_min_var = aligned_returns.dot(weights_min_var)

# Portfólio Tangente (usando a linha de mercado de capitais)
excess_returns = aligned_returns.mean() - rf
weights_tangent = inv_cov_matrix.dot(excess_returns) / ones.dot(inv_cov_matrix).dot(excess_returns)
returns_tangent = aligned_returns.dot(weights_tangent)

# Estatísticas dos portfólios
portfolios = {
    'ingênuo': returns_ine,
    'mínima variância': returns_min_var,
    'tangente': returns_tangent
}

portfolio_stats = pd.DataFrame(index=portfolios.keys(), columns=['mean', 'median', 'std', 'variance', 'sharpe', 'beta', 'treynor', 'var'])

for name, port_ret in portfolios.items():
    portfolio_stats.at[name, 'mean'] = port_ret.mean() * 252
    portfolio_stats.at[name, 'median'] = port_ret.median()
    portfolio_stats.at[name, 'std'] = port_ret.std() * np.sqrt(252)
    portfolio_stats.at[name, 'variance'] = port_ret.var() * 252
    portfolio_stats.at[name, 'sharpe'] = (port_ret.mean() - rf) / port_ret.std() * np.sqrt(252)
    portfolio_stats.at[name, 'beta'] = np.cov(port_ret, aligned_ibov_returns)[0, 1] / aligned_ibov_returns.var()
    portfolio_stats.at[name, 'treynor'] = (port_ret.mean() - rf) / portfolio_stats.at[name, 'beta']
    portfolio_stats.at[name, 'var'] = port_ret.mean() - z * port_ret.std()

print(portfolio_stats)

# Salvar as estatísticas e os portfólios em arquivos XLS
with pd.ExcelWriter('financial_analysis_results.xlsx') as writer:
    data.to_excel(writer, sheet_name='Preços Históricos')
    returns.to_excel(writer, sheet_name='Retornos')
    stats.to_excel(writer, sheet_name='Estatísticas')
    portfolio_stats.to_excel(writer, sheet_name='Portfólios')

# Fronteira Eficiente

def portfolio_performance(weights, returns):
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_stddev

def negative_sharpe_ratio(weights, returns, risk_free_rate):
    p_return, p_stddev = portfolio_performance(weights, returns)
    return -(p_return - risk_free_rate) / p_stddev

def minimize_variance(weights):
    return portfolio_performance(weights, returns)[1]

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(len(tickers)))

# Inicializar pesos
init_guess = len(tickers) * [1. / len(tickers), ]

# Portfólio de Mínima Variância
opt_results = minimize(minimize_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
min_var_weights = opt_results.x

# Portfólio de Tangência
opt_results = minimize(negative_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, args=(aligned_returns, rf))
tangent_weights = opt_results.x

# Plotar a Fronteira Eficiente
def plot_efficient_frontier(returns):
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    port_returns = []
    port_stddevs = []
    
    for _ in range(10000):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        port_return, port_stddev = portfolio_performance(weights, returns)
        port_returns.append(port_return)
        port_stddevs.append(port_stddev)
    
    port_returns = np.array(port_returns)
    port_stddevs = np.array(port_stddevs)
    
    plt.scatter(port_stddevs, port_returns, c=(port_returns - rf) / port_stddevs, marker='o')
    plt.xlabel('Volatilidade')
    plt.ylabel('Retorno')
    plt.colorbar(label='Sharpe Ratio')
    
    plt.scatter(np.sqrt(np.dot(min_var_weights.T, np.dot(cov_matrix, min_var_weights))), np.sum(min_var_weights * mean_returns), marker='*', color='r', s=100, label='Portfólio de Mínima Variância')
    plt.scatter(np.sqrt(np.dot(tangent_weights.T, np.dot(cov_matrix, tangent_weights))), np.sum(tangent_weights * mean_returns), marker='*', color='g', s=100, label='Portfólio Tangente')
    plt.scatter(np.sqrt(np.dot(weights_ine.T, np.dot(cov_matrix, weights_ine))), np.sum(weights_ine * mean_returns), marker='*', color='b', s=100, label='Portfólio Ingênuo')
    
    plt.legend()
    plt.show()

plot_efficient_frontier(aligned_returns)
