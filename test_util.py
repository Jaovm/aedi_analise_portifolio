from util import *

def test_download_finance_data():
    """
    Test the download_finance_data function
    """
    tickers = "AAPL"
    start = "2022-01-01"
    end = "2022-12-31"
    dados = download_finance_data(tickers, start, end)
    assert not dados.empty, "Data should not be empty"
    assert "Close" in dados.columns, "Data should contain 'Close' column"
    
def test_get_valid_tickers_and_normalized_weights():
    """
    Test the get_valid_tickers_and_normalized_weights function
    """
    input_tickers = ["AAPL", "GOOGL", "", "MSFT"]
    input_weights = [0.4, 0.3, 0.1, 0.2]
    valid_tickers, normalized_weights = get_valid_tickers_and_normalized_weights(input_tickers, input_weights)
    
    assert valid_tickers == ["AAPL", "GOOGL", "MSFT"], "Valid tickers should be ['AAPL', 'GOOGL', 'MSFT']"
    assert len(normalized_weights) == 3, "There should be 3 normalized weights"
    assert sum(normalized_weights) == 1, "Normalized weights should sum to 1"
    assert all(isinstance(weight, float) for weight in normalized_weights), "All weights should be floats"

def test_generate_price_history_fig():
    """
    Test the generate_price_history_fig function
    """
    # Create sample data for stocks
    data_yf_stocks = download_finance_data(['AAPL'], '2024-06-03', '2024-06-07')['Adj Close']
    # Create sample data for index
    data_yf_index = download_finance_data(['^BVSP'], '2024-06-03', '2024-06-07')['Adj Close']

    # Generate the figure
    fig = generate_price_history_fig(data_yf_stocks, data_yf_index)

    # Check if the figure has the correct number of traces
    assert len(fig.data) == 2, "Figure should have 2 traces (1 for stocks and 1 for index)"

    # Check if the subplot titles are correct
    assert fig.layout.annotations[0].text == 'Histórico de Preços dos Ativos', "First subplot title should be 'Histórico de Preços dos Ativos'"
    assert fig.layout.annotations[1].text == 'Histórico do Índice Bovespa', "Second subplot title should be 'Histórico do Índice Bovespa'"

    # Check if the x-axis and y-axis titles are correct
    assert fig.layout.xaxis2.title.text == 'Período', "X-axis title should be 'Período'"
    assert fig.layout.yaxis.title.text == 'Preço de Fechamento Ajustado', "First y-axis title should be 'Preço de Fechamento Ajustado'"
    assert fig.layout.yaxis2.title.text == 'Fechamento Ajustado', "Second y-axis title should be 'Fechamento Ajustado'"
