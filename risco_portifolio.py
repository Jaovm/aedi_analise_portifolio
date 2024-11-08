"""
Utilização de Simulações de Monte Carlo para Análise de Risco e Retorno de Portifólio de Ações

Utiliza streamlit para interface gráfica.


Autor: Fernando sola Pereira
"""
import random

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import plotly.express as px

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuração da página
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.header('Parâmetros')

# Horizonte de tempo (em dias)
horizon = st.sidebar.text_input('Horizonte de Tempo (dias)', 30)

# Graus de liberdade da distribuição t de Student
degrees_freedom = st.sidebar.text_input('Graus de Liberdade', 5)

# Nível de confiança para o VaR
confidence_level = st.sidebar.text_input('Nível de Confiança', 95)

# Número de simulações de Monte Carlo
n_simulations = st.sidebar.text_input('Número de Simulações', 1000)

# Estabelecer um limite de 300000 na relação entre o produto de horizonte e o número de simulações
if int(horizon) * int(n_simulations) * int(degrees_freedom) > 20000000:
    st.sidebar.error("O produto entre Horizonte de Tempo, Graus de Liberdade e Número de Simulações não pode exceder 20.000.000 Por favor, ajuste os valores.")
    st.stop()

# Título da página
st.title('Análise de Risco e Retorno de Portifólio de Ações')

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
s_tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 'BBAS3.SA']
s_weights = [1.0] * 6

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

# documentar o processo em markdown
md = """

## Introdução

A Simulação de Monte Carlo é uma técnica utilizada para modelar sistemas complexos e incertos, permitindo a \
análise de resultados em diferentes cenários aleatórios. Neste projeto, utiliza-se a simulação de Monte Carlo \
para analisar o risco e retorno de um portifólio de ações. Assim, foi escolhida a distribuição t Student para
estimar o Value at Risk (VaR) de um portifólio de ações.

## Fundamentação

A distribuição t de Student é uma distribuição de probabilidade contínua que surge quando se estima a média \
de uma população normalmente distribuída, mas a variância populacional é desconhecida e substituída pela \
variância amostral. Ela é particularmente útil em amostras de pequeno tamanho, onde a incerteza sobre a \
variância populacional é maior.

Matematicamente, a distribuição t de Student com __𝜈__ graus de liberdade é definida pela função de densidade de \
probabilidade:"""

col_dados.markdown(md)

latex_code = r"""
f(t) = \frac{\Gamma \left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu \pi} \Gamma \left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}
"""
col_dados.latex(latex_code)


md = """\\
onde __Γ__ é a função gama e __𝜈__ representa os graus de liberdade.

Em análises financeiras, o modelo de distribuição normal é frequentemente usado para representar os retornos \
de ativos. Contudo, dados reais mostram que esses retornos geralmente têm "caudas pesadas", ou seja, eventos \
extremos (grandes perdas ou ganhos) acontecem com mais frequência do que o previsto pela curva normal.

A distribuição t de Student é uma alternativa melhor nesse caso, pois acomoda essas caudas pesadas, capturando \
melhor a chance de eventos extremos. Isso leva a estimativas de risco mais precisas, especialmente para métricas \
como o VaR, que são influenciadas por esses eventos.

O VaR é uma medida estatística que quantifica a perda potencial máxima esperada de um portfólio \
em um determinado horizonte de tempo, para um dado nível de confiança. Assim, considerando-se um VaR de -0,50 \
com 95% de confiança para 365 dias, por exemplo, significa que há 95% de confiança de que a perda não excederá \
50% do valor do portfólio ao longo dos próximos 365 dias. Da mesma forma, há uma probabilidade de 5% de que a \
perda seja superior a 50% nesse período.


## Metodologia

Para realizar a análise de risco e retorno do portifólio de ações, foram seguidos os seguintes passos:

1. Definição dos parâmetros da simulação: 
    * Horizonte de Tempo: número de dias para o cálculo dos retornos acumulados da carteira.
    * Graus de liberdade: Graus de liberdade da distribuição t de Student
    * Nível de confiança para o VaR 
    * Número de simulações de Monte Carlo.

2. Coleta dos dados históricos dos ativos: os preços de fechamento ajustados dos ativos foram baixados do Yahoo \
Finance para o período especificado.

3. Cálculo dos retornos diários dos ativos: os retornos diários são calculados com base nos preços de fechamento \
ajustados.

4. Estimação dos parâmetros da distribuição t de Student: para cada ativo, foram calculados o retorno médio diário \
e a volatilidade média diária.

5. Simulação de Monte Carlo: são realizadas simulações de Monte Carlo para gerar cenários de retornos futuros \
para cada ativo, com base na distribuição t de Student.

6. Cálculo dos retornos diários da carteira: os retornos diários da carteira foram calculados como a soma dos retornos \
diários dos ativos, ponderados pelos pesos especificados.

7. Cálculo dos retornos acumulados da carteira: os retornos acumulados da carteira para o horizonte de tempo \
especificado foram calculados.

8. Cálculo do VaR: o VaR para o horizonte de tempo especificado foi calculado com base na distribuição dos retornos \
acumulados da carteira.

9. Análise dos resultados: os resultados foram apresentados em termos de VaR e distribuição dos retornos acumulados \
da carteira.

10. A simulação também é feita utilizando-se uma normal permitindo a comparação dos resultados de ambas as distribuições.

## Resultados

A principal diferença observada ao utilizar a distribuição t de Student é o aumento da probabilidade de eventos \
extremos devido às suas caudas mais pesadas. Isso resulta em um VaR mais conservador (ou seja, uma perda potencial \
maior) em comparação com a distribuição normal. No contexto da gestão de riscos, isso significa que o modelo está \
levando em consideração a maior chance de ocorrerem perdas significativas, proporcionando uma estimativa de \
risco mais realista.
"""
col_dados.markdown(md)

   
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
for ticker, weight in zip(valid_tickers, normalized_weights):
    dados[ticker]['Retorno'] = dados[ticker]['Adj Close'].pct_change()
    dados[ticker] = dados[ticker].dropna()
    retornos[ticker] = {
        'Retorno Médio Diário': dados[ticker]['Retorno'].mean(),
        'Volatilidade Média Diária': dados[ticker]['Retorno'].std(),
        'Peso Normalizado': weight
    }

# Criar um DataFrame com os resultados
resultados = pd.DataFrame(retornos).T.reset_index().rename(columns={'index': 'Ticker'})

# Exibir o DataFrame na coluna de dados sem o índice
col_graficos.write(resultados)

# Parâmetros da distribuição t de Student para os retornos dos ativos
n_s = int(n_simulations)
n_h = int(horizon)

simulated_returns_t = []
simulated_returns_normal = []

for i, ticker in enumerate(valid_tickers):
    loc = retornos[ticker]['Retorno Médio Diário']
    scale = retornos[ticker]['Volatilidade Média Diária']
    peso = retornos[ticker]['Peso Normalizado']

    # simular com normal
    simulated_returns_normal.append(peso * np.random.normal(loc=loc, scale=scale, size=(n_s, n_h)))

    # simular com t-Student
    df = int(degrees_freedom)
    simulated_returns_t.append(peso * t.rvs(df=df, loc=loc, scale=scale, size=(n_s, n_h)))



# Cálculo dos retornos diários da carteira
portfolio_returns = np.sum(simulated_returns_t, axis=0)
cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1
VaR = np.percentile(cumulative_returns, 100 - float(confidence_level))
col_graficos.markdown(f'VaR - t de Student ({confidence_level}% de confiança) para {horizon} dias: __{VaR:.4%}__')

# calculo com a simulacao da normal
portfolio_returns_normal = np.sum(simulated_returns_normal, axis=0)
cumulative_returns_normal = np.prod(1 + portfolio_returns_normal, axis=1) - 1
VaR_normal = np.percentile(cumulative_returns_normal, 100 - float(confidence_level))
col_graficos.markdown(f'VaR - Normal ({confidence_level}% de confiança) para {horizon} dias: __{VaR_normal:.4%}__')

# Histograma dos retornos acumulados da carteira com t-Student e Normal
fig = px.histogram(
    pd.DataFrame({'t de Student':cumulative_returns, 'Normal':cumulative_returns_normal}), 
    nbins=200, 
    opacity=0.5, 
    labels={'value': 'Retorno Acumulado da Carteira'}, 
    title=f'Distribuição dos Retornos da Carteira ({horizon} dias)',
)

fig.update_layout(
    xaxis_title='Retorno Acumulado da Carteira', 
    yaxis_title='Frequência', 
    showlegend=True,
    legend=dict(title='Distribuição', itemsizing='constant'),
)

fig.add_vline(x=VaR, line_width=3, line_dash="dash", line_color="green", annotation_text='VaR t-Student', annotation_position="top left")
fig.add_vline(x=VaR_normal, line_width=3, line_dash="dash", line_color="red", annotation_text='VaR Normal', annotation_position="top right")

col_graficos.plotly_chart(fig)