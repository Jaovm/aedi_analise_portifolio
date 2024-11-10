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


# semente aleatória fixada para reprodutividade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# utilizar o espaço todo do container
st.set_page_config(layout="wide")

# sidebar
st.sidebar.header('Parâmetros')

col5, col6 = st.sidebar.columns(2)

with col5:
    # Horizonte de tempo (em dias)
    horizon = st.text_input('Horizonte de Tempo (dias)', 30)
    # Número de simulações de Monte Carlo
    n_simulations = st.text_input('Número de Simulações', 1000)

with col6:
    # Graus de liberdade da distribuição t de Student
    degrees_freedom = st.text_input('Graus de Liberdade', 5)
    # Nível de confiança para o VaR
    confidence_level = st.text_input('Nível de Confiança', 95)


# Estabelecer um limite de 300000 na relação entre o produto de horizonte e o número de simulações
LIMITE_SIMULACOES = 2000000000
if int(horizon) * int(n_simulations) * int(degrees_freedom) > LIMITE_SIMULACOES:
    st.sidebar.error(f"O produto entre Horizonte de Tempo, Graus de Liberdade e Número de Simulações não pode exceder {LIMITE_SIMULACOES} Por favor, ajuste os valores.")
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
s_tickers = ['TAEE4.SA', 'VALE3.SA', 'VIVT3.SA', 'BBSE3.SA', 'BBAS3.SA']
s_weights = [1.0] * len(s_tickers)

input_tickers = []
input_weights = []
for i in range(len(s_tickers)):
    with col1:
        ticker = st.text_input(f'Ticker do Ativo {i+1}', s_tickers[i])
        input_tickers.append(ticker)
    with col2:
        weight = st.text_input(f'Peso do Ativo {i+1}', f'{s_weights[i]:.4}')
        input_weights.append(weight)

# Filtrar tickers e pesos válidos
valid_tickers = [ticker for ticker in input_tickers if ticker]
valid_weights = [float(input_weights[i]) for i in range(len(input_tickers)) if input_tickers[i]]

# Normalizar os pesos para somarem 1
total_weight = sum(valid_weights)
normalized_weights = [weight / total_weight for weight in valid_weights]

# Baixar os dados históricos
dados = {}
for ticker in valid_tickers:
    dados[ticker] = yf.download(ticker, start=inicio, end=fim)

dados['^BVSP'] = yf.download('^BVSP', start=inicio, end=fim)

container = st.container()

# Realize uma análise gráfica descritiva temporal do preço das ações: Você deve
# plotar os dados de preços das acoes ao longo do tempo para cada uma das acoes
# selecionadas e para o ındice. Visualize como o preco das acoes mudou ao longo
# do tempo e identifique possıveis tendencias.

# Plotar os preços de fechamento ajustados dos ativos usando plotly express, exceto do ticker ^BVSP
# plotar 2 gráficos, um com as ações e o outro com o índice da B3 (^BVSP). Eles devem possuir os eixos X compartilhados
# para preservar a mesma linha de tempo

# titulo
container.markdown('## Histórico de Preços dos Ativos e do Índice Bovespa')
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Histórico de Preços dos Ativos', 'Histórico do Índice Bovespa'))

fig1 = px.line(
    pd.concat([dados[ticker]['Adj Close'] for ticker in valid_tickers], axis=1),
    labels={'value': 'Preço de Fechamento Ajustado', 'index': 'Data', 'variable': 'Ativo'},
    title='Preço de Fechamento Ajustado dos Ativos',
)

for d in fig1.data:
    fig.add_trace(d, row=1, col=1)

fig2 = px.line(
    dados['^BVSP']['Adj Close'],
    labels={'value': 'Preço de Fechamento Ajustado', 'index': 'Data'},
    title='Fechamento Ajustado do Índice Bovespa',
)

for d in fig2.data:
    fig.add_trace(d, row=2, col=1)

fig.update_xaxes(title_text="Período", row=2, col=1)
fig.update_yaxes(title_text="Preço de Fechamento Ajustado", row=1, col=1)
fig.update_yaxes(title_text="Fechamento Ajustado", row=2, col=1)

container.plotly_chart(fig)

# Você deve calcular os retornos diários das ações e do índice e plotar os dados de retorno ao longo do tempo 
# para cada uma das ações e para o índice. 
retornos = {}
dados['^BVSP']['Retorno'] = dados['^BVSP']['Adj Close'].pct_change()
for ticker, weight in zip(valid_tickers, normalized_weights):
    dados[ticker]['Retorno'] = dados[ticker]['Adj Close'].pct_change()
    retornos[ticker] = {
        'Retorno Médio Diário': dados[ticker]['Retorno'].mean(),
        'Volatilidade Média Diária': dados[ticker]['Retorno'].std(),
        'Peso Normalizado': weight
    }

# Visualize como o retorno das ações mudou ao longo do tempo e identifique possíveis padrões.
container.markdown('## Retornos Diários dos Ativos e do Índice Bovespa')

df_retornos = pd.concat([dados[ticker]['Retorno'] for ticker in valid_tickers], axis=1)
df_retornos.columns = valid_tickers
df_retornos['^BVSP'] = dados['^BVSP']['Retorno']
                        
fig = px.line(
    df_retornos,
    labels={'value': 'Retorno Diário', 'index': 'Data', 'variable': 'Ativo'},
    title='',
)

container.plotly_chart(fig)


# Correlações entre os retornos diários
container.markdown('## Correlação entre os Retornos Diários')

# Calcular a matriz de correlação dos retornos diários
correlacao = df_retornos.corr()

# Definindo o esquema de cores com vermelho em valores altos (próximo a 1)
custom_colorscale = [
    [0.0, 'green'],   # Cor para o limite inferior (-1)
    [0.5, 'blue'],  # Cor para o valor neutro (0)
    [1.0, 'red']     # Cor para o limite superior (1)
]

# Criando o heatmap com o esquema de cores customizado
fig = px.imshow(correlacao, 
                text_auto=True,  
                aspect="auto",   
                color_continuous_scale=custom_colorscale,  # Esquema de cores customizado
                labels=dict(color="Correlações"),
                zmin=-1, zmax=1)  # Definindo o limite de correlações

container.plotly_chart(fig)


##########################################################################################
# Calcular Fronteira Eficiente de Markowitz
##########################################################################################
container.markdown('## Fronteira Eficiente de Markowitz')
df_fronteira = pd.concat([dados[t]['Adj Close'] for t in valid_tickers], axis=1)
df_fronteira.columns = valid_tickers
returns = df_fronteira.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 4. Simulação de Portfólios Aleatórios
num_portfolios = 50000
results = np.zeros((3, num_portfolios))
all_weights = np.zeros((num_portfolios, len(valid_tickers)))

for i in range(num_portfolios):
    # Gerar pesos aleatórios
    weights = np.random.random(len(valid_tickers))
    weights /= np.sum(weights)

    # Calcula o retorno e risco do portfólio
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Armazena os resultados
    results[0,i] = portfolio_std_dev
    results[1,i] = portfolio_return
    results[2,i] = results[1,i] / results[0,i]

    # Armazena os pesos
    all_weights[i, :] = weights

# 5. Visualização da Fronteira Eficiente
results_frame = pd.DataFrame(results.T, columns=['Risco', 'Retorno', 'Sharpe'])
for i, ticker in enumerate(valid_tickers):
    results_frame[ticker + '_weight'] = all_weights[:, i]

# 6. Encontrar os Portfólios Ótimos
max_sharpe_idx = results_frame['Sharpe'].idxmax()
max_sharpe_port = results_frame.iloc[max_sharpe_idx]
min_risk_idx = results_frame['Risco'].idxmin()
min_risk_port = results_frame.iloc[min_risk_idx]


# Criar o gráfico base com go.Scatter
fig = px.scatter(
    x=results_frame['Risco'], 
    y=results_frame['Retorno'], 
    color=results_frame['Sharpe'], 
    color_continuous_scale='viridis',
)

# Adicionar o ponto do portfólio Máx. Sharpe
fig.add_trace(go.Scatter(
    x=[max_sharpe_port['Risco']], 
    y=[max_sharpe_port['Retorno']], 
    mode='markers',
    marker=dict(color='red', size=20, symbol='circle-open-dot', opacity=1, line=dict(width=2, color='black')), 
    name='Máx. Sharpe',
    showlegend=True
))

# Adicionar o ponto do portfólio Mín. Risco
fig.add_trace(go.Scatter(
    x=[min_risk_port['Risco']], 
    y=[min_risk_port['Retorno']], 
    mode='markers', 
    marker=dict(color='gray', size=20, symbol='circle-open-dot', opacity=1, line=dict(width=2, color='black')), 
    name='Mín. Risco', 
    showlegend=True
))

# Configurar o layout do gráfico, incluindo a posição da legenda
fig.update_layout(
    title='Fronteira Eficiente de Markowitz - Simulação de Monte Carlo',
    xaxis_title='Risco (Desvio Padrão)',
    yaxis_title='Retorno Esperado',
    coloraxis_colorbar_title='Sharpe Ratio',
    height=600,
    legend=dict(
        x=1,  # Posição horizontal da legenda (1 = direita)
        y=1,  # Posição vertical da legenda (1 = topo)
        xanchor='right',  # Ancorar a legenda à direita
        yanchor='top'   # Ancorar a legenda ao topo
    ),
    coloraxis_colorbar=dict(title="Sharpe Ratio"),
)

container.plotly_chart(fig)



##########################################################################################
# Portifólio
##########################################################################################
container.markdown('## Portifólios Otimizados')

print("Pesos do Portfólio de Máximo Sharpe Ratio:")
print("Pesos do Portfólio de Mínimo Risco:")

portifolio = []
for i, ticker in enumerate(valid_tickers):
    print(f"{ticker}: {max_sharpe_port[ticker + '_weight']:.2%}")
    print(f"{ticker}: {min_risk_port[ticker + '_weight']:.2%}")
    portifolio.append({
        'Ativo': ticker,
        'Retorno Médio Diário': dados[ticker]['Retorno'].mean(),
        'Volatilidade Média Diária': dados[ticker]['Retorno'].std(),
        'Peso Informado (Normalizado)': retornos[ticker]['Peso Normalizado'],
        'Max Sharpe': max_sharpe_port[ticker + '_weight'],
        'Min Risco': min_risk_port[ticker + '_weight'],
    })

# Criar um DataFrame com os resultados
resultados = pd.DataFrame(portifolio).set_index('Ativo')

# Use st.markdown com HTML e CSS para centralizar
container.markdown(f'''<div style="display: flex; justify-content: center;"></div>''', unsafe_allow_html=True)
container.write(resultados)
container.markdown(f'''</div>''', unsafe_allow_html=True)


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
container.markdown(f'VaR - t de Student ({confidence_level}% de confiança) para {horizon} dias: __{VaR:.4%}__')

# calculo com a simulacao da normal
portfolio_returns_normal = np.sum(simulated_returns_normal, axis=0)
cumulative_returns_normal = np.prod(1 + portfolio_returns_normal, axis=1) - 1
VaR_normal = np.percentile(cumulative_returns_normal, 100 - float(confidence_level))
container.markdown(f'VaR - Normal ({confidence_level}% de confiança) para {horizon} dias: __{VaR_normal:.4%}__')

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

container.plotly_chart(fig)


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

container.markdown(md)

latex_code = r"""
f(t) = \frac{\Gamma \left( \frac{\nu + 1}{2} \right)}{\sqrt{\nu \pi} \Gamma \left( \frac{\nu}{2} \right)} \left( 1 + \frac{t^2}{\nu} \right)^{-\frac{\nu + 1}{2}}
"""
container.latex(latex_code)


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
container.markdown(md)