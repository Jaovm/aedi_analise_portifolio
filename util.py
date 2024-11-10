"""

"""

# native python libs
import random

from typing import List

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

def download_finance_data(tickers: str | List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download finance data from Yahoo Finance
    """
    dados = yf.download(tickers, start=start, end=end, repair=False)
    return dados


def get_valid_tickers_and_normalized_weights(input_tickers, input_weights):
    valid_tickers = [ticker for ticker in input_tickers if ticker]
    valid_weights = [float(input_weights[i]) for i in range(len(input_tickers)) if input_tickers[i]]

    # Normalizar os pesos para somarem 1
    total_weight = sum(valid_weights)
    normalized_weights = [weight / total_weight for weight in valid_weights]
    return valid_tickers,normalized_weights


def generate_price_history_fig(data_yf_stocks, data_yf_index) -> go.Figure:
    """
    Generate a plotly figure with the historical prices of the assets and the Bovespa index
    """

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[2,1])

    fig1 = px.line(
        data_yf_stocks,
        labels={'value': 'Preço de Fechamento Ajustado', 'index': 'Data', 'variable': 'Ativo'},
        title='Preço de Fechamento Ajustado dos Ativos',
    )

    for d in fig1.data:
        fig.add_trace(d, row=1, col=1)

    fig2 = px.line(
        data_yf_index,
        labels={'value': 'Preço de Fechamento Ajustado', 'index': 'Data'},
        title='Fechamento Ajustado do Índice Bovespa',
    )

    for d in fig2.data:
        fig.add_trace(d, row=2, col=1)

    fig.update_xaxes(title_text="Período", row=2, col=1)
    fig.update_yaxes(title_text="Preço de Fechamento Ajustado", row=1, col=1)
    fig.update_yaxes(title_text="Fechamento Ajustado", row=2, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_layout(
        height=700,
        title_text="Histórico de Preços dos Ativos e do Índice Bovespa",
        showlegend=True,
    )    

    return fig


def generate_returns_plot(df_retornos) -> go.Figure:
    fig = px.line(
        df_retornos,
        labels={'value': 'Retorno Diário', 'index': 'Data', 'variable': 'Ativo'},
    )
    
    return fig


def generate_correlation_plot(df_retornos: pd.DataFrame) -> go.Figure:
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
    
    return fig


def optimize_portfolio_allocation(valid_tickers, daily_returns_stocks, num_portfolios):
    
    returns = daily_returns_stocks.dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Simulação de Portfólios Aleatórios
    num_portfolios = num_portfolios
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
    return results_frame,max_sharpe_port,min_risk_port


def generate_portfolio_risk_return_plot(results_frame, max_sharpe_port, min_risk_port):
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
    
    return fig


def generate_portfolio_summary(valid_tickers, normalized_weights, daily_returns_stocks, max_sharpe_port, min_risk_port):
    portifolio = []
    for i, ticker in enumerate(valid_tickers):
        portifolio.append({
        'Ativo': ticker,
        'Retorno Médio Diário': f'{daily_returns_stocks[ticker].mean():.4f}',
        'Volatilidade Média Diária': f'{daily_returns_stocks[ticker].std():.4f}',
        'Peso Informado (Normalizado)': f'{normalized_weights[i]:.2%}',
        'Melhor Sharpe': f"{max_sharpe_port[ticker + '_weight']:.2%}",
        'Menor Risco': f"{min_risk_port[ticker + '_weight']:.2%}",
    })

    # Criar um DataFrame com os resultados
    resultados = pd.DataFrame(portifolio).set_index('Ativo')
    return resultados


def generate_return_simulations(horizon, n_simulations, degrees_freedom, confidence_level, valid_tickers, normalized_weights, daily_returns_stocks):
    # Parâmetros da distribuição t de Student para os retornos dos ativos
    n_s = int(n_simulations)
    n_h = int(horizon)

    simulated_returns_t = []
    simulated_returns_normal = []

    for i, ticker in enumerate(valid_tickers):
        loc = daily_returns_stocks[ticker].mean()
        scale = daily_returns_stocks[ticker].std()
        peso = normalized_weights[i]

        # simular com normal
        simulated_returns_normal.append(peso * np.random.normal(loc=loc, scale=scale, size=(n_s, n_h)))

        # simular com t-Student
        df = int(degrees_freedom)
        simulated_returns_t.append(peso * t.rvs(df=df, loc=loc, scale=scale, size=(n_s, n_h)))



    # Cálculo dos retornos diários da carteira
    portfolio_returns = np.sum(simulated_returns_t, axis=0)
    cumulative_returns = np.prod(1 + portfolio_returns, axis=1) - 1
    VaR = np.percentile(cumulative_returns, 100 - float(confidence_level))

    # calculo com a simulacao da normal
    portfolio_returns_normal = np.sum(simulated_returns_normal, axis=0)
    cumulative_returns_normal = np.prod(1 + portfolio_returns_normal, axis=1) - 1
    VaR_normal = np.percentile(cumulative_returns_normal, 100 - float(confidence_level))

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
    return fig,VaR,VaR_normal