"""
Utilização de Simulações de Monte Carlo para Análise de Risco e Retorno de Portifólio de Ações

Utiliza streamlit para interface gráfica.


Autor: Fernando sola Pereira
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import plotly.express as px


# Configuração da página
st.set_page_config(layout="wide")
st.title('Análise de Risco e Retorno de Portifólio de Ações')

# Sidebar
st.sidebar.header('Parâmetros')

# Horizonte de tempo (em dias)
horizon = st.sidebar.text_input('Horizonte de Tempo (dias)', 30)

# Graus de liberdade da distribuição t de Student
degrees_freedom = st.sidebar.text_input('Graus de Liberdade', 5)

# Nível de confiança para o VaR
confidence_level = st.sidebar.text_input('Nível de Confiança', 95)

# Número de simulações de Monte Carlo
n_simulations = st.sidebar.text_input('Número de Simulações', 10000)


# Título da seção de dados
st.sidebar.markdown('## Período para o Histórico')

# Período de análise dos dados históricos
col3, col4 = st.sidebar.columns(2)

with col3:
    inicio = st.text_input('Data de Início', '2010-01-01')

with col4:
    fim = st.text_input('Data de Fim', '2024-10-31')

# Título da seção de dados
st.sidebar.markdown('## Dados dos Ativos')

# Ticker e peso dos ativos
col1, col2 = st.sidebar.columns(2)

#colocar 6 tickers das principais ações da B3
s_tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'B3SA3.SA']
s_weights = [1/6] * 6

tickers = []
weights = []
for i in range(6):
    with col1:
        ticker = st.text_input(f'Ticker do Ativo {i+1}', s_tickers[i-1])
        tickers.append(ticker)
    with col2:
        weight = st.text_input(f'Peso do Ativo {i+1}', f'{s_weights[i-1]:.4}')
        weights.append(weight)


# definir layout em 2 colunas dados, graficos
container = st.container()
col_dados, col_graficos = container.columns(2)

   
# Filtrar tickers e pesos válidos
valid_tickers = [ticker for ticker in tickers if ticker]
valid_weights = [float(weights[i]) for i in range(len(tickers)) if tickers[i]]

# Normalizar os pesos para somarem 1
total_weight = sum(valid_weights)
normalized_weights = [weight / total_weight for weight in valid_weights]

# Baixar os dados históricos
dados = {}
for ticker in valid_tickers:
    dados[ticker] = yf.download(ticker, start=inicio, end=fim)

# Calcular os retornos diários para cada ativo
retornos = {}
for ticker in valid_tickers:
    dados[ticker]['Retorno'] = dados[ticker]['Adj Close'].pct_change()
    dados[ticker] = dados[ticker].dropna()
    retornos[ticker] = {
        'Retorno Médio Diário': dados[ticker]['Retorno'].mean(),
        'Volatilidade Média Diária': dados[ticker]['Retorno'].std()
    }

# Criar um DataFrame com os resultados
resultados = pd.DataFrame(retornos).T.reset_index().rename(columns={'index': 'Ticker'})

# Exibir o DataFrame na coluna de dados sem o índice
col_dados.write(resultados)

# Parâmetros da distribuição t de Student para os retornos dos ativos
n_s = int(n_simulations)
n_h = int(horizon)
simulated_returns = []

for i, ticker in enumerate(valid_tickers):
    df = int(degrees_freedom)
    loc = retornos[ticker]['Retorno Médio Diário']
    scale = retornos[ticker]['Volatilidade Média Diária']
    simulated_returns.append(normalized_weights[i] * t.rvs(df=df, loc=loc, scale=scale, size=(n_s, n_h)))

# Cálculo dos retornos diários da carteira
portfolio_returns = np.sum(simulated_returns, axis=0)

# Cálculo dos retornos acumulados da carteira para o horizonte de tempo
cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1

# Cálculo do VaR (95% de confiança)
VaR = np.percentile(cumulative_returns, 100 - float(confidence_level))

# Impressão do resultado
col_dados.write(f'VaR ({confidence_level}% de confiança) para {horizon} dias: {VaR:.4f}')

# Histograma dos retornos acumulados da carteira

fig = px.histogram(
    cumulative_returns, 
    nbins=200, 
    labels={'value': 'Retorno Acumulado da Carteira'}, 
    title=f'Distribuição dos Retornos da Carteira ({horizon} dias)'
)

fig.update_layout(xaxis_title='Retorno Acumulado da Carteira', yaxis_title='Frequência')    
col_graficos.plotly_chart(fig)

