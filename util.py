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

from scipy.optimize import minimize
from scipy.stats import t
from plotly.subplots import make_subplots

# semente aleatória fixada para reprodutividade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# outras constantes
CONSTANTE_ANUALIZACAO = 252


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


def generate_efficient_frontier_points(daily_returns_stocks, annualized_returns=True) -> List:

    returns = daily_returns_stocks.dropna()
    mean_returns = returns.mean()

    cov_matrix = returns.cov()

    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.dot(weights, mean_returns)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std

    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    # 4. Função para Otimizar Portfólio para Retorno Alvo
    def efficient_portfolio(mean_returns, cov_matrix, target_return):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}
        )
        bounds = tuple((0, 1) for asset in range(num_assets))
        result = minimize(minimize_volatility, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    # 5. Gerar a Fronteira Eficiente
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 200)
    efficient_portfolios = []
    for target_return in target_returns:
        portfolio = efficient_portfolio(mean_returns, cov_matrix, target_return)
        if portfolio.success:
            returns, std = portfolio_performance(portfolio.x, mean_returns, cov_matrix)

            if annualized_returns:
                returns = returns * CONSTANTE_ANUALIZACAO
                std = std * np.sqrt(CONSTANTE_ANUALIZACAO)

            efficient_portfolios.append({'Retorno': returns, 'Volatilidade': std, 'Pesos': portfolio.x})

    return efficient_portfolios    


def optimize_portfolio_allocation(valid_tickers, daily_returns_stocks, num_portfolios, annualized_returns=True):
    
    returns = daily_returns_stocks.dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    if annualized_returns:
        mean_returns = mean_returns * CONSTANTE_ANUALIZACAO
        cov_matrix = cov_matrix * CONSTANTE_ANUALIZACAO

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


def generate_portfolio_risk_return_plot(results_frame, max_sharpe_port, min_risk_port, daily_returns_stocks, valid_tickers, normalized_weights, annualized_returns=True):
    # Criar o gráfico base com go.Scatter
    fig = px.scatter(
        x=results_frame['Risco'], 
        y=results_frame['Retorno'], 
        color=results_frame['Sharpe'], 
        color_continuous_scale='viridis',
        opacity=0.9,
    )

    efficient_frontier_points = generate_efficient_frontier_points(daily_returns_stocks, annualized_returns)
    fig.add_trace(go.Scattergl(
        x=np.asarray([point['Volatilidade'] for point in efficient_frontier_points]), 
        y=np.asarray([point['Retorno'] for point in efficient_frontier_points]),
        name='Fronteira Eficiente',
        mode='lines',
        opacity=0.7,
        line=dict(width=3, color='gray', dash='dash'),
        showlegend=False
    ))

    # criar um ponto para cada média (y), desvio (x) dos ativos
    drs = daily_returns_stocks.dropna()
    means = drs.mean()[valid_tickers]
    stds = drs.std()[valid_tickers]

    # Adicionar o ponto do portfólio informado
    normalized_weights = np.asarray(normalized_weights)
    cov_matrix = drs.cov()
    portfolio_return = np.dot(normalized_weights, means)
    portfolio_std_dev = np.sqrt(np.dot(normalized_weights.T, np.dot(cov_matrix, normalized_weights)))

    if annualized_returns:
        portfolio_return = portfolio_return * CONSTANTE_ANUALIZACAO
        portfolio_std_dev = portfolio_std_dev * np.sqrt(CONSTANTE_ANUALIZACAO)
    
    fig.add_trace(go.Scattergl(
        x=[portfolio_std_dev], 
        y=[portfolio_return], 
        mode='markers+text', 
        marker=dict(size=15, opacity=1, line=dict(width=1, color='gray'), color='white', symbol='hourglass'), 
        text=['Portifolio Informado'], 
        textposition='top center',
        showlegend=False
    ))

    # Adicionar o ponto do portfólio Máx. Sharpe
    fig.add_trace(go.Scattergl(
        x=[max_sharpe_port['Risco']], 
        y=[max_sharpe_port['Retorno']], 
        mode='markers+text',
        marker=dict(size=15, opacity=1, line=dict(width=1, color='gray'), color='darkgreen', symbol='arrow-bar-down'), 
        name='Melhor Sharpe',
        text=['Melhor Sharpe'],
        textposition='top left',
        showlegend=False
    ))

    # Adicionar o ponto do portfólio Mín. Risco
    fig.add_trace(go.Scattergl(
        x=[min_risk_port['Risco']], 
        y=[min_risk_port['Retorno']], 
        mode='markers+text', 
        marker=dict(size=15, opacity=1, line=dict(width=1, color='gray'), color='darkgreen', symbol='arrow-bar-right'), 
        name='Menor Risco',
        text=['Menor Risco'], 
        textposition='top left',
        showlegend=False
    ))

    if annualized_returns:
        means = means * CONSTANTE_ANUALIZACAO
        stds = stds * np.sqrt(CONSTANTE_ANUALIZACAO)

    for mean, std, ticker in zip(means, stds, drs.columns):
        fig.add_trace(go.Scattergl(
            x=[std], 
            y=[mean], 
            mode='markers+text', 
            text=[ticker],
            name=ticker,
            textposition='top center',
            marker=dict(size=15, symbol='x', opacity=1, line=dict(width=1, color='gray')), 
            showlegend=False
        ))

    # Configurar o layout do gráfico, incluindo a posição da legenda
    fig.update_layout(
        xaxis_title='Risco',
        yaxis_title='Retorno Esperado',
        yaxis_tickformat='.4f',
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
        'Retorno Médio Diário': f'{daily_returns_stocks[ticker].mean():.5f}',
        'Volatilidade Média Diária': f'{daily_returns_stocks[ticker].std():.5f}',
        'Peso Informado (Normalizado)': f'{normalized_weights[i]:.2%}',
        'Melhor Sharpe': f"{max_sharpe_port[ticker + '_weight']:.2%}",
        'Menor Risco': f"{min_risk_port[ticker + '_weight']:.2%}",
    })

    # Criar um DataFrame com os resultados
    resultados = pd.DataFrame(portifolio).set_index('Ativo')
    return resultados

def calcular_var_carteira(retornos, pesos, horizonte=1, simulacoes=10000, nivel_confianca=0.95, graus_liberdade=5):
    """
    Calcula o Value at Risk (VaR) de uma carteira de ativos usando simulação de Monte Carlo.

    Args:
        retornos (pd.DataFrame): DataFrame com os retornos diários dos ativos.
        pesos (np.array): Array com os pesos de cada ativo na carteira.
        horizonte (int): Horizonte de tempo em dias para o cálculo do VaR (padrão: 1 dia).
        simulacoes (int): Número de simulações de Monte Carlo (padrão: 10000).
        nivel_confianca (float): Nível de confiança para o cálculo do VaR (padrão: 0.95).

    Returns:
        float: Value at Risk (VaR) em percentual.
    """
    # Calcular retornos da carteira
    retornos_carteira = (retornos * pesos).sum(axis=1)

    # Calcular média e desvio padrão dos retornos da carteira
    media = retornos_carteira.mean()
    desvio_padrao = retornos_carteira.std()

    # Gerar retornos logarítmicos aleatórios para a carteira
    np.random.seed(42)  # Para reprodutibilidade
    retornos_simulados = np.random.normal(media, desvio_padrao, (horizonte, simulacoes))

    retornos_simulados_t = np.random.standard_t(graus_liberdade, (horizonte, simulacoes))
    retornos_simulados_t = retornos_simulados_t * desvio_padrao + media

    # Calcular retornos logarítmicos acumulados
    retornos_acumulados = retornos_simulados.cumsum(axis=0)
    retornos_acumulados_t = retornos_simulados_t.cumsum(axis=0)

    # Simular trajetórias de preços da carteira (assumindo valor inicial de 1)
    caminhos_precos = np.exp(retornos_acumulados)
    caminhos_precos_t = np.exp(retornos_acumulados_t)

    # Calcular o retorno percentual
    precos_finais = caminhos_precos[-1]
    retornos_percentuais = (precos_finais - 1)

    precos_finais_t = caminhos_precos_t[-1]
    retornos_percentuais_t = (precos_finais_t - 1)

    # Calcular o Value at Risk (VaR) em percentual
    VaR_percentual = np.percentile(retornos_percentuais, (1 - nivel_confianca) * 100)
    VaR_percentual_t = np.percentile(retornos_percentuais_t, (1 - nivel_confianca) * 100)

    # retornos esperados (esperança da simulação)
    retornos_percentuais_esperados = retornos_percentuais.mean()
    retornos_percentuais_esperados_t = retornos_percentuais_t.mean()

    return VaR_percentual, VaR_percentual_t, retornos_percentuais, retornos_percentuais_t, retornos_percentuais_esperados, retornos_percentuais_esperados_t

def generate_return_simulations(horizon, n_simulations, degrees_freedom, confidence_level, valid_tickers, portifolio_weights, daily_returns_stocks, annualized_returns=True):
    # # Parâmetros da distribuição t de Student para os retornos dos ativos
    n_simulations = int(n_simulations)
    horizon = int(horizon)
    confidence_level = float(confidence_level) / 100
    degrees_freedom = float(degrees_freedom)

    VaR_percentual, VaR_percentual_t, retornos_percentuais, retornos_percentuais_t, retornos_percentuais_esperados, retornos_percentuais_esperados_t = calcular_var_carteira(daily_returns_stocks, portifolio_weights.values, horizon, n_simulations, confidence_level, degrees_freedom)

    # Histograma dos retornos acumulados da carteira com t-Student e Normal
    fig = px.histogram(
        pd.DataFrame({'t de Student':retornos_percentuais_t, 'Normal':retornos_percentuais}), 
        nbins=200, 
        opacity=0.5, 
        labels={'value': 'Retorno Acumulado da Carteira'}, 
        title=f'Distribuição dos Retornos da Carteira ({horizon} {"anos" if annualized_returns else "dias"})',
    )

    fig.update_layout(
        xaxis_title='Retorno Acumulado da Carteira', 
        yaxis_title='Frequência',
        showlegend=True,
        legend=dict(title='Distribuição', itemsizing='constant'),
    )

    fig.add_vline(x=VaR_percentual_t, line_width=3, line_dash="dash", line_color="green", annotation_text='VaR t-Student', annotation_position="top left")
    fig.add_vline(x=VaR_percentual, line_width=3, line_dash="dash", line_color="red", annotation_text='VaR Normal', annotation_position="top right")
    return fig, VaR_percentual_t, VaR_percentual, retornos_percentuais_esperados_t, retornos_percentuais_esperados