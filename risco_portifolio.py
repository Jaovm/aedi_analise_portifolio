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

# # Definir o ticker da ação (por exemplo, 'PETR4.SA' para Petrobras)
# ticker_a = 'BBSE3.SA'
# ticker_b = 'BBAS3.SA'

# # Definir o período de análise
# inicio = '2010-01-01'
# fim = '2024-10-31'

# # Baixar os dados históricos
# dados_a = yf.download(ticker_a, start=inicio, end=fim)
# dados_b = yf.download(ticker_b, start=inicio, end=fim)

# # Calcular os retornos diários para dados_a
# dados_a['Retorno'] = dados_a['Adj Close'].pct_change()
# dados_a = dados_a.dropna()
# retorno_medio_diario_a = dados_a['Retorno'].mean()
# volatilidade_diaria_a = dados_a['Retorno'].std()

# # Calcular os retornos diários para dados_b
# dados_b['Retorno'] = dados_b['Adj Close'].pct_change()
# dados_b = dados_b.dropna()
# retorno_medio_diario_b = dados_b['Retorno'].mean()
# volatilidade_diaria_b = dados_b['Retorno'].std()


# print(f"Retorno médio diário da ação {ticker_a} entre {inicio} e {fim}: {retorno_medio_diario_a:.5%}")
# print(f"Volatilidade diária da ação {ticker_a}: {volatilidade_diaria_a:.5%}")

# print(f"Retorno médio diário da ação {ticker_b} entre {inicio} e {fim}: {retorno_medio_diario_b:.5%}")
# print(f"Volatilidade diária da ação {ticker_b}: {volatilidade_diaria_b:.5%}")

# # Parâmetros da distribuição t de Student para os retornos dos ativos
# df_A = 5  # Graus de liberdade do ativo A
# loc_A = retorno_medio_diario_a  # Média do retorno diário do ativo A
# scale_A = volatilidade_diaria_b  # Desvio padrão do retorno diário do ativo A
# df_B = 5  # Graus de liberdade do ativo B
# loc_B = retorno_medio_diario_b  # Média do retorno diário do ativo B
# scale_B = volatilidade_diaria_b  # Desvio padrão do retorno diário do ativo B

# # Pesos da carteira
# w_A = 0  # Peso do ativo A na carteira
# w_B = 1  # Peso do ativo B na carteira

# # Número de simulações de Monte Carlo
# n_simulations = 10000

# # Horizonte de tempo (em dias)
# horizon = 30

# # Simulação dos retornos diários dos ativos
# returns_A = t.rvs(df=df_A, loc=loc_A, scale=scale_A, size=(n_simulations, horizon))
# returns_B = t.rvs(df=df_B, loc=loc_B, scale=scale_B, size=(n_simulations, horizon))

# # Cálculo dos retornos diários da carteira
# portfolio_returns = w_A * returns_A + w_B * returns_B

# # Cálculo dos retornos acumulados da carteira para o horizonte de tempo
# cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1

# # Cálculo do VaR (95% de confiança)
# VaR = np.percentile(cumulative_returns, 5)

# # Impressão do resultado
# print(f'VaR (95% de confiança) para {horizon} dias: {VaR:.4f}')

# # Histograma dos retornos acumulados da carteira
# plt.hist(cumulative_returns, bins=50)
# plt.xlabel('Retorno Acumulado da Carteira')
# plt.ylabel('Frequência')
# plt.title(f'Distribuição dos Retornos da Carteira ({horizon} dias)')
# plt.show()

# criar um layout com 2 panels um à esquerda e um à direita. o da esquerda terá parâmetros 
# e campos para inserção de dados e o da direita terá gráficos e resultados
st.set_page_config(layout="wide")
st.title('Simulações de Monte Carlo para Análise de Risco e Retorno de Portifólio de Ações')

st.sidebar.header('Parâmetros')

# # botao para executar a simulação
# simular = st.sidebar.button('Simular')

# Horizonte de tempo (em dias)
horizon = st.sidebar.text_input('Horizonte de Tempo (dias)', 30)


# Definir os graus de liberdade da distribuição t de Student
degrees_freedom = st.sidebar.text_input('Graus de Liberdade', 5)

# Definir o nível de confiança para o VaR
confidence_level = st.sidebar.text_input('Nível de Confiança', 95)

# Definir o número de simulações de Monte Carlo
n_simulations = st.sidebar.text_input('Número de Simulações', 10000)

# Definir o ticker da ação A
ticker_a = st.sidebar.text_input('Ticker do Ativo A', 'PETR4.SA')
# Definir o peso do ativo A na carteira
weight_a = st.sidebar.text_input('Peso do Ativo A', 0.5)

# Definir o ticker da ação B
ticker_b = st.sidebar.text_input('Ticker do Ativo B', 'VALE3.SA')
# Definir o peso do ativo B na carteira
weight_b = st.sidebar.text_input('Peso do Ativo B', 0.5)

# Definir o período de análise
inicio = st.sidebar.text_input('Data de Início', '2010-01-01')
fim = st.sidebar.text_input('Data de Fim', '2024-10-31')

# definir layout em 2 colunas dados, graficos
container = st.container()
col_dados, col_graficos = container.columns(2)

# Criar um botão que atualiza a simulação
def executar_simulacao():
    
    # Baixar os dados históricos
    dados_a = yf.download(ticker_a, start=inicio, end=fim)
    dados_b = yf.download(ticker_b, start=inicio, end=fim)

    # Calcular os retornos diários para dados_a
    dados_a['Retorno'] = dados_a['Adj Close'].pct_change()
    dados_a = dados_a.dropna()
    retorno_medio_diario_a = dados_a['Retorno'].mean()
    volatilidade_diaria_a = dados_a['Retorno'].std()

    # Calcular os retornos diários para dados_b
    dados_b['Retorno'] = dados_b['Adj Close'].pct_change()
    dados_b = dados_b.dropna()
    retorno_medio_diario_b = dados_b['Retorno'].mean()
    volatilidade_diaria_b = dados_b['Retorno'].std()

    col_dados.write(f"Retorno médio diário da ação {ticker_a} entre {inicio} e {fim}: {retorno_medio_diario_a:.5%}")

    col_dados.write(f"Volatilidade diária da ação {ticker_a}: {volatilidade_diaria_a:.5%}")

    col_dados.write(f"Retorno médio diário da ação {ticker_b} entre {inicio} e {fim}: {retorno_medio_diario_b:.5%}")

    col_dados.write(f"Volatilidade diária da ação {ticker_b}: {volatilidade_diaria_b:.5%}")

    # Parâmetros da distribuição t de Student para os retornos dos ativos
    df_A = int(degrees_freedom)  # Graus de liberdade do ativo A
    loc_A = retorno_medio_diario_a  # Média do retorno diário do ativo A
    scale_A = volatilidade_diaria_a  # Desvio padrão do retorno diário do ativo A
    df_B = 5  # Graus de liberdade do ativo B
    loc_B = retorno_medio_diario_b  # Média do retorno diário do ativo B
    scale_B = volatilidade_diaria_b  # Desvio padrão do retorno diário do ativo B

    # Simulação dos retornos diários dos ativos
    n_s = int(n_simulations)
    n_h = int(horizon)

    returns_A = t.rvs(df=df_A, loc=loc_A, scale=scale_A, size=(n_s, n_h))
    returns_B = t.rvs(df=df_B, loc=loc_B, scale=scale_B, size=(n_s, n_h))

    # Cálculo dos retornos diários da carteira
    w_a = float(weight_a)
    w_b = float(weight_b)
    portfolio_returns = w_a * returns_A + w_b * returns_B

    # Cálculo dos retornos acumulados da carteira para o horizonte de tempo
    cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1

    # Cálculo do VaR (95% de confiança)
    VaR = np.percentile(cumulative_returns, 5)

    # Impressão do resultado
    col_dados.write(f'VaR (95% de confiança) para {horizon} dias: {VaR:.4f}')

    # Histograma dos retornos acumulados da carteira
    fig, ax = plt.subplots()
    ax.hist(cumulative_returns, bins=50)
    ax.set_xlabel('Retorno Acumulado da Carteira')
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição dos Retornos da Carteira ({horizon} dias)')
    col_graficos.pyplot(fig)


executar_simulacao()