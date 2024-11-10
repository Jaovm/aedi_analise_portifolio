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
st.sidebar.markdown('## Dados dos Ativos')

# Ticker e peso dos ativos
col1, col2 = st.sidebar.columns(2)

#colocar 6 tickers das principais ações da B3
s_tickers = ['EGIE3.SA', 'VALE3.SA', 'VIVT3.SA', 'BBSE3.SA', 'BBAS3.SA']
s_weights = [0.2] * len(s_tickers)

input_tickers = []
input_weights = []
for i in range(len(s_tickers)):
    with col1:
        ticker = st.text_input(f'Ticker do Ativo {i+1}', s_tickers[i])
        input_tickers.append(ticker)
    with col2:
        weight = st.text_input(f'Peso do Ativo {i+1}', f'{s_weights[i]:.4}')
        input_weights.append(weight)


#################################################
# Seção de configuração do VaR
#################################################
st.sidebar.markdown('## Value at Risk (VaR)')

# opção anualizado ou diário
anualizado = st.sidebar.radio('Frequência do VaR', ['Anual', 'Diário'])

col5, col6 = st.sidebar.columns(2)

with col5:
    # Horizonte de tempo (em anos ou dias)

    horizon = st.text_input(f'Horizonte de Tempo ({anualizado})', 5)
    # Número de simulações de Monte Carlo
    n_simulations = st.text_input('Número de Simulações', 20000)

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
    inicio = st.text_input('Data de Início', '2020-01-01')

with col4:
    fim = st.text_input('Data de Fim', '2024-10-31')


#################################################
# Seção de Fronteira Eficiente de Markowitz
#################################################
# Título da seção de dados
st.sidebar.markdown('## Fronteira Eficiente de Markowitz')

# Período de análise dos dados históricos
numero_carteiras_fem = st.sidebar.text_input('Número de Carteiras Simuladas', 50000)

#################################################
# Seção de Investimento
#################################################
aporte_inicial = st.sidebar.text_input('Aporte Inicial (R$)', 35000)


#################################################
# Processamentos
#################################################
annualized_returns = anualizado == 'Anual'
valid_tickers, normalized_weights = util.get_valid_tickers_and_normalized_weights(input_tickers, input_weights)

# Baixar os dados históricos
dados = {}
data_yf_stocks = util.download_finance_data(valid_tickers, start=inicio, end=fim)['Adj Close']
data_yf_index = util.download_finance_data('^BVSP', start=inicio, end=fim)['Adj Close']

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
daily_returns_stocks = data_yf_stocks.pct_change()
daily_returns_index = data_yf_index.pct_change()
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

        st.markdown(f'__t de Student ({confidence_level}% de confiança) para {horizon} {"anos" if anualizado == "Anual" else "dias"}__')
        st.markdown(f'VaR: __{VaR:.4%}__')
        st.markdown(f'Esperança de Retorno: __{mean_portfolio_return:.4%}__')
        ve = mean_portfolio_return * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'Valor Esperado: __R$ {ve}__')

        st.markdown(f'__Normal ({confidence_level}% de confiança) para {horizon} {"anos" if anualizado == "Anual" else "dias"}__')
        st.markdown(f'VaR: __{VaR_normal:.4%}__')
        st.markdown(f'Esperança de Retorno: __{mean_portfolio_return_normal:.4%}__')
        ve = mean_portfolio_return_normal * float(aporte_inicial) + float(aporte_inicial)
        ve = f'{ve:,.2f}'.replace(',', 'v').replace('.', ',').replace('v', '.')
        st.markdown(f'Valor Esperado: __R$ {ve}__')

        st.plotly_chart(fig)


# documentar o processo em markdown
md = r"""
## Introdução

A Simulação de Monte Carlo é uma técnica utilizada para modelar sistemas complexos e incertos, permitindo a 
análise de resultados em diferentes cenários aleatórios. Neste projeto, utiliza-se a simulação de Monte Carlo 
para analisar o risco e retorno de um portifólio de ações. Assim, foi escolhida a distribuição t Student para 
estimar o Value at Risk (VaR) de um portifólio de ações.

## Fundamentação

A distribuição t de Student é uma distribuição de probabilidade contínua que surge quando se estima a média 
de uma população normalmente distribuída, mas a variância populacional é desconhecida e substituída pela 
variância amostral. Ela é particularmente útil em amostras de pequeno tamanho, onde a incerteza sobre a 
variância populacional é maior.

Matematicamente, a distribuição t de Student com __𝜈__ graus de liberdade é definida pela função de densidade de 
probabilidade:

$f(t) = \frac{\Gamma \left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu \pi} \Gamma \left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}$

onde __Γ__ é a função gama e __𝜈__ representa os graus de liberdade.

Em análises financeiras, o modelo de distribuição normal é frequentemente usado para representar os retornos
de ativos. Contudo, dados reais mostram que esses retornos geralmente têm "caudas pesadas", ou seja, eventos
extremos (grandes perdas ou ganhos) acontecem com mais frequência do que o previsto pela curva normal.

A distribuição t de Student é uma alternativa melhor nesse caso, pois acomoda essas caudas pesadas, capturando 
melhor a chance de eventos extremos. Isso leva a estimativas de risco mais precisas, especialmente para métricas 
como o VaR, que são influenciadas por esses eventos.

O VaR é uma medida estatística que quantifica a perda potencial máxima esperada de um portfólio 
em um determinado horizonte de tempo, para um dado nível de confiança. Assim, considerando-se um VaR de -0,50 
com 95% de confiança para 365 dias, por exemplo, significa que há 95% de confiança de que a perda não excederá 
50% do valor do portfólio ao longo dos próximos 365 dias. Da mesma forma, há uma probabilidade de 5% de que a 
perda seja superior a 50% nesse período.


## Metodologia

Para realizar a análise de risco e retorno do portifólio de ações, foram seguidos os seguintes passos:

1. Definição dos parâmetros da simulação: 
    * Horizonte de Tempo: número de dias para o cálculo dos retornos acumulados da carteira.
    * Graus de liberdade: Graus de liberdade da distribuição t de Student
    * Nível de confiança para o VaR 
    * Número de simulações de Monte Carlo.

2. Coleta dos dados históricos dos ativos: os preços de fechamento ajustados dos ativos foram baixados do Yahoo 
Finance para o período especificado.

3. Cálculo dos retornos diários dos ativos: os retornos diários são calculados com base nos preços de fechamento 
ajustados.

4. Estimação dos parâmetros da distribuição t de Student: para cada ativo, foram calculados o retorno médio diário 
e a volatilidade média diária.

5. Simulação de Monte Carlo: são realizadas simulações de Monte Carlo para gerar cenários de retornos futuros 
para cada ativo, com base na distribuição t de Student.

6. Cálculo dos retornos diários da carteira: os retornos diários da carteira foram calculados como a soma dos retornos 
diários dos ativos, ponderados pelos pesos especificados.

7. Cálculo dos retornos acumulados da carteira: os retornos acumulados da carteira para o horizonte de tempo 
especificado foram calculados.

8. Cálculo do VaR: o VaR para o horizonte de tempo especificado foi calculado com base na distribuição dos retornos 
acumulados da carteira.

9. Análise dos resultados: os resultados foram apresentados em termos de VaR e distribuição dos retornos acumulados 
da carteira.

10. A simulação também é feita utilizando-se uma normal permitindo a comparação dos resultados de ambas as distribuições.

## Resultados

A principal diferença observada ao utilizar a distribuição t de Student é o aumento da probabilidade de eventos 
extremos devido às suas caudas mais pesadas. Isso resulta em um VaR mais conservador (ou seja, uma perda potencial 
maior) em comparação com a distribuição normal. No contexto da gestão de riscos, isso significa que o modelo está 
levando em consideração a maior chance de ocorrerem perdas significativas, proporcionando uma estimativa de 
risco mais realista.
"""
container.markdown(md)