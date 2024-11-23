"""
Utilização de Simulações de Monte Carlo para Análise de Risco e Retorno de Portifólio de Ações

Utiliza streamlit para interface gráfica.


Autor: Fernando sola Pereira
"""
# native python libs
import random

# third-party libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import t
from plotly.subplots import make_subplots

# local modules
import util


# semente aleatória fixada para reprodutividade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# outras variaveis globais
LIMITE_SIMULACOES = 20000000

# anualizado
annualized_returns = False

# utilizar o espaço todo do container
st.set_page_config(layout="wide")


# título da página
st.title('Análise de Risco e Retorno de Portifólio de Ações')

# título sidebar
st.sidebar.header('Parâmetros')

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)



#################################################
# Seção de configuração de ativos
#################################################

# Ticker e peso dos ativos
col1, col2, col3 = st.sidebar.columns([3, 1, 1])
col1.markdown('## Dados dos Ativos')

# inserir dois botões para adicionar ou remover ativos
if st.session_state.get('s_tickers') is None:
    #colocar 6 tickers das principais ações da B3
    s_tickers = ['WEGE3.SA', 'VALE3.SA', 'SUZB3.SA', 'EGIE3.SA', 'TRPL4.SA']
    s_weights = [1.0] * len(s_tickers)

    st.session_state.s_tickers = s_tickers
    st.session_state.s_weights = s_weights


# incluir os botoes na coluna 4 da sidebar
add_button = col2.button('( + )')
remove_button = col3.button('( - )')

# se clicar no botão de adicionar, adicionar um novo ticker e peso
if add_button:
    st.session_state.s_tickers.append('')
    st.session_state.s_weights.append(1.0)

# se clicar no botão de remover, remover o último ticker e peso
if remove_button:
    if len(st.session_state.s_tickers) > 0:
        st.session_state.s_tickers.pop()
        st.session_state.s_weights.pop()

col1, col2 = st.sidebar.columns(2)

input_tickers = []
input_weights = []
for i in range(len(st.session_state.s_tickers)):
    with col1:
        ticker = st.text_input(f'Ticker do Ativo {i+1}', st.session_state.s_tickers[i])
        input_tickers.append(ticker)
    with col2:
        weight = st.text_input(f'Peso do Ativo {i+1}', f'{st.session_state.s_weights[i]:.4}')
        input_weights.append(weight)


#################################################
# Seção de configuração do VaR
#################################################
st.sidebar.markdown('## Value at Risk (VaR)')

col5, col6 = st.sidebar.columns(2)

with col5:
    # Horizonte de tempo (em anos ou dias)

    horizon = st.text_input(f'Horizonte de Tempo (D)', 1)
    # Número de simulações de Monte Carlo
    n_simulations = st.text_input('Número de Simulações', 1000)

with col6:
    # Graus de liberdade da distribuição t de Student
    degrees_freedom = st.text_input('Graus de Liberdade', 5)
    # Nível de confiança para o VaR
    confidence_level = st.text_input('Nível de Confiança', 95)


# Estabelecer um limite de 300000 na relação entre o produto de horizonte e o número de simulações
if int(horizon) * int(n_simulations) * int(degrees_freedom) > LIMITE_SIMULACOES:
    st.sidebar.error(f"O produto entre Horizonte de Tempo, Graus de Liberdade e Número de Simulações não pode exceder {LIMITE_SIMULACOES} Por favor, ajuste os valores.")
    st.stop()


#################################################
# Seção de configuração de período histórico
#################################################
# Título da seção de dados
st.sidebar.markdown('## Período para o Histórico')

# Período de análise dos dados históricos
col3, col4 = st.sidebar.columns(2)

with col3:
    inicio = st.text_input('Data de Início', '2018-01-01')

with col4:
    fim = st.text_input('Data de Fim', '2023-12-31')


#################################################
# Seção de Fronteira Eficiente de Markowitz
#################################################
# Título da seção de dados
st.sidebar.markdown('## Fronteira Eficiente de Markowitz')

# Período de análise dos dados históricos
numero_carteiras_fem = st.sidebar.text_input('Número de Carteiras Simuladas', 10000)

#################################################
# Seção de Investimento
#################################################
st.sidebar.markdown('## Investimento')
aporte_inicial = st.sidebar.text_input('Aporte Inicial (R$)', 35000)


#################################################
# Processamentos
#################################################
valid_tickers, normalized_weights = util.get_valid_tickers_and_normalized_weights(input_tickers, input_weights)

# Baixar os dados históricos
dados = {}
data_yf_stocks = util.download_finance_data(valid_tickers, start=inicio, end=fim)
data_yf_stocks = data_yf_stocks['Adj Close']
data_yf_stocks = data_yf_stocks[valid_tickers]

data_yf_index = util.download_finance_data('^BVSP', start=inicio, end=fim)
data_yf_index = data_yf_index['Adj Close']
container = st.container()

container = st.container()

# Realize uma análise gráfica descritiva temporal do preço das ações: Você deve
# plotar os dados de preços das acoes ao longo do tempo para cada uma das acoes
# selecionadas e para o ındice. Visualize como o preco das acoes mudou ao longo
# do tempo e identifique possıveis tendencias.

# Plotar os preços de fechamento ajustados dos ativos usando plotly express, exceto do ticker ^BVSP
# plotar 2 gráficos, um com as ações e o outro com o índice da B3 (^BVSP). Eles devem possuir os eixos X compartilhados
# para preservar a mesma linha de tempo

container.markdown('## Histórico de Preços dos Ativos e do Índice Bovespa')
fig = util.generate_price_history_fig(data_yf_stocks, data_yf_index)
container.plotly_chart(fig)

# Você deve calcular os retornos diários das ações e do índice e plotar os dados de retorno ao longo do tempo 
# para cada uma das ações e para o índice. 
# daily_returns_stocks = data_yf_stocks.pct_change().dropna()
daily_returns_stocks = np.log(data_yf_stocks / data_yf_stocks.shift(1)).dropna()

# daily_returns_index = data_yf_index.pct_change().dropna()
daily_returns_index = np.log(data_yf_index / data_yf_index.shift(1)).dropna()
df_retornos = pd.concat([daily_returns_stocks, daily_returns_index], axis=1)

# Visualize como o retorno das ações mudou ao longo do tempo e identifique possíveis padrões.
container.markdown('## Retornos Diários dos Ativos e do Índice Bovespa')
fig = util.generate_returns_plot(df_retornos)
container.plotly_chart(fig)

# Correlações entre os retornos diários
container.markdown('## Correlação entre os Retornos Diários')
fig = util.generate_correlation_plot(df_retornos)
container.plotly_chart(fig)


##########################################################################################
# Calcular Fronteira Eficiente de Markowitz
##########################################################################################
container.markdown('## Fronteira Eficiente de Markowitz')
num_portifolios = int(numero_carteiras_fem)
results_frame, max_sharpe_port, min_risk_port = util.optimize_portfolio_allocation(valid_tickers, daily_returns_stocks, num_portifolios, annualized_returns)

# incluir os risco/reetorno dos ativos isolados
fig = util.generate_portfolio_risk_return_plot(results_frame, max_sharpe_port, min_risk_port, daily_returns_stocks, valid_tickers, normalized_weights, annualized_returns)
container.plotly_chart(fig)



##########################################################################################
# Portifólio
##########################################################################################
container.markdown('## Portifólios Otimizados')
df_resultados = util.generate_portfolio_summary(valid_tickers, normalized_weights, daily_returns_stocks, max_sharpe_port, min_risk_port)

# Use st.markdown com HTML e CSS para centralizar
container.markdown(f'''<div style="display: flex; justify-content: center;"></div>''', unsafe_allow_html=True)
container.write(df_resultados)
container.markdown(f'''</div>''', unsafe_allow_html=True)


# gerar distribuição VaR para a carteira informada, a de menor risco e a de melhor sharper

portifolios = []
for i, ticker in enumerate(valid_tickers):
    portifolios.append({
    'Ativo': ticker,
    'Pesos Informados': normalized_weights[i],
    'Menor Risco': min_risk_port[ticker + '_weight'],
    'Melhor Sharpe': max_sharpe_port[ticker + '_weight'],
})
    
df_portifolios = pd.DataFrame(portifolios).loc[:, 'Pesos Informados':]

for portifolio, coluna in zip(df_portifolios.columns, container.columns(3)):
    fig,VaR,VaR_normal, mean_portfolio_return, mean_portfolio_return_normal = util.generate_return_simulations(
        horizon, 
        n_simulations, 
        degrees_freedom, 
        confidence_level, 
        valid_tickers, 
        df_portifolios[portifolio], 
        daily_returns_stocks, 
        annualized_returns=annualized_returns
        )
    with coluna:
        st.markdown(f'#### {portifolio}')

        st.markdown(f'##### __t de Student ({confidence_level}% de confiança) para {horizon} {"anos" if annualized_returns else "dias"}__')
        st.markdown(f'* VaR: __{VaR:.4%}__')
        ve = VaR * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'* Valor Esperado na Perda máxima: __{ve}__')
        st.markdown(f'* Esperança de Retorno: __{mean_portfolio_return:.4%}__')
        ve = mean_portfolio_return * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'* Valor Esperado: __R$ {ve}__')

        st.markdown(f'##### __Normal ({confidence_level}% de confiança) para {horizon} {"anos" if annualized_returns else "dias"}__')
        st.markdown(f'* VaR: __{VaR_normal:.4%}__')
        ve = VaR_normal * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'* Valor Esperado na Perda máxima: __{ve}__')
        st.markdown(f'* Esperança de Retorno: __{mean_portfolio_return_normal:.4%}__')
        ve = mean_portfolio_return_normal * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'* Valor Esperado: __R$ {ve}__')

        st.plotly_chart(fig)