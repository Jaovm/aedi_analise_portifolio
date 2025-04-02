import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Função para baixar os dados históricos dos ativos
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Função para calcular o retorno diário e a matriz de covariância
def calculate_statistics(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

# Função de otimização da carteira (Markowitz)
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0):
    num_assets = len(mean_returns)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def portfolio_return(weights):
        return np.dot(weights, mean_returns)
    def negative_sharpe_ratio(weights):
        p_return = portfolio_return(weights)
        p_volatility = portfolio_volatility(weights)
        return -(p_return - risk_free_rate) / p_volatility

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Função para gerar a simulação da carteira
def simulate_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = (portfolio_return - 0.01) / portfolio_volatility  # Sharpe ratio
    return results

# Streamlit interface
st.title('Otimização de Carteira de Investimentos com Teoria de Markowitz')

# Input para o usuário selecionar ativos
tickers = st.text_input("Insira os tickers dos ativos (separados por vírgula)", "AAPL,MSFT,GOOG,AMZN")
tickers = tickers.split(',')

start_date = st.date_input("Data de Início", pd.to_datetime("2019-01-01"))
end_date = st.date_input("Data de Término", pd.to_datetime("2025-01-01"))

# Baixar dados dos ativos
data = get_data(tickers, start_date, end_date)
returns = data.pct_change().dropna()

mean_returns, cov_matrix = calculate_statistics(returns)

# Otimização da carteira
result = optimize_portfolio(mean_returns, cov_matrix)
st.write("Pesos ótimos da carteira:", result.x)

# Simulação de portfólios
num_portfolios = 10000
results = simulate_portfolios(num_portfolios, mean_returns, cov_matrix)

# Exibir os resultados
st.write(f"Melhor Sharpe Ratio: {results[2].max()}")
st.write(f"Retorno ótimo: {results[0][results[2].argmax()]}")
st.write(f"Volatilidade ótima: {results[1][results[2].argmax()]}")

# Gráfico de dispersão
fig, ax = plt.subplots()
ax.scatter(results[1], results[0], c=results[2], cmap='YlGnBu', marker='o')
ax.set_xlabel('Volatilidade')
ax.set_ylabel('Retorno')
st.pyplot(fig)
