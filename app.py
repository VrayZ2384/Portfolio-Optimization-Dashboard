import streamlit as st
from tabs.portfolio_overview import portfolio_overview
from tabs.stock_performance import stock_performance
from tabs.portfolio_allocation import app
from tabs.market_data import market_data
from tabs.financial_ratios import financial_ratios
from tabs.portfolio_metrics import portfolio_metrics
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(layout="wide")

def read_css(file_path):
    with open(file_path, 'r') as f:
        return f'<style>{f.read()}</style>'


css_content = read_css('static/styles.css')
st.markdown(css_content, unsafe_allow_html=True)

# Streamlit layout with columns for title and About button
st.markdown('<div class="title-container"><h1>Investment Portfolio Dashboard</h1>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
about_button = st.button("About")
back_button_top = st.button("Back to Dashboard")

if about_button:
    st.markdown("""
        # About This Application
        This Investment Portfolio Dashboard is designed to provide a comprehensive analysis of your investment portfolio.
        It includes the following features:

        ## Portfolio Overview
        - **Descriptive Statistics**: This section provides basic statistics for each stock in the portfolio, including count, mean, standard deviation, min, max, and quartiles. These statistics give a quick summary of the distribution and central tendency of the stock prices.
        - **Total Portfolio Value**: Displays the total value of the portfolio by summing the last available prices of all stocks. This gives an overview of the overall worth of the portfolio.
        - **Daily Returns**: Calculates and plots the daily percentage change in stock prices. This indicates the day-to-day performance and volatility of each stock in the portfolio.
        - **Cumulative Returns**: Shows the cumulative growth of each stock since the start date. This is useful for understanding the overall growth trend and long-term performance of the stocks in the portfolio.
        - **Performance Metrics**:
          - **Annualized Return**: Provides an estimate of the portfolio's return over a year, helping in comparing the performance of the portfolio with annual benchmarks.
          - **Annualized Volatility**: Offers an understanding of the risk associated with the portfolio by estimating the expected fluctuation in returns over a year.
          - **Sharpe Ratio**: Measures the risk-adjusted return, indicating how much return is received for the extra volatility endured. A higher Sharpe ratio suggests better risk-adjusted performance.
          - **Max Drawdown**: Measures the largest observed loss from a peak to a trough in the value of the portfolio, indicating the maximum observed loss. This is a crucial metric for understanding potential downside risk.
        - **Drawdown Visualization**: Plots the drawdown for each stock, showing how much each stock has fallen from its peak value over time. This helps in visualizing the risk of significant losses in the portfolio.
        - **Correlation Matrix**: Shows the correlation coefficients between the returns of each pair of stocks. A correlation coefficient of 1 means the stocks move perfectly together, -1 means they move inversely, and 0 means no correlation. This matrix helps in understanding the diversification benefit in the portfolio.

        ## Stock Performance
        The `stock_performance` module provides detailed performance metrics and visualizations for individual stocks. It includes:
        - **Price Trends**: Visualization of historical price data to understand how the stock price has moved over time. This can help in identifying long-term trends and patterns.
        - **Volume Trends**: Analysis of trading volume over time, which can indicate the level of interest in a stock and potential market sentiment changes.
        - **Moving Averages**: Calculation and plotting of various moving averages (e.g., 50-day, 200-day) to help identify the direction of the trend and potential buy or sell signals.

        ## Portfolio Allocation
        The `portfolio_allocation` module helps in visualizing and optimizing the allocation of assets in your portfolio. It includes:
        - **Current Allocation**: Visualization of the current asset allocation, showing the proportion of the portfolio invested in each stock. This helps in understanding the distribution of investments.
        - **Suggested Allocation**: Optimization algorithms to suggest the best asset allocation based on various criteria, such as risk tolerance and investment goals. This can help in maximizing returns while minimizing risk.

        ## Market Data
        The `market_data` module provides access to up-to-date market data and trends. It includes:
        - **Stock Data Retrieval**: Fetches historical and real-time stock data to keep the portfolio analysis current and relevant.
        - **Market Indicators**: Analysis of key market indicators such as indices (e.g., S&P 500, Dow Jones) and sector performance. This can help in understanding the broader market context and how it impacts individual stocks.

        ## Financial Ratios
        The `financial_ratios` module evaluates stocks using key financial ratios. It includes:
        - **PE Ratio (Price-to-Earnings Ratio)**: Indicates how much investors are willing to pay per dollar of earnings. A higher PE ratio might indicate that a stock is overvalued, or that investors are expecting high growth rates in the future.
        - **PB Ratio (Price-to-Book Ratio)**: Compares a company's market value to its book value. A lower PB ratio might indicate that the stock is undervalued.
        - **ROE (Return on Equity)**: Measures a company's profitability relative to shareholders' equity. A higher ROE indicates more efficient use of equity.
        - **ROA (Return on Assets)**: Indicates how profitable a company is relative to its total assets. A higher ROA shows that the company is efficient at generating profit from its assets.
        - **Dividend Yield**: Shows how much a company pays out in dividends each year relative to its stock price. This is important for income-focused investors.

        ## Portfolio Metrics
        The `portfolio_metrics` module provides advanced metrics to assess portfolio performance and risk. It includes:
        - **Value at Risk (VaR)**: A statistical technique used to measure the risk of loss of an investment. It estimates how much a set of investments might lose, given normal market conditions, in a set time period.
        - **Conditional VaR (CVaR)**: The average loss that occurs beyond the VaR threshold. It provides an estimate of the tail risk.
        - **Monte Carlo Simulation**: A simulation technique used to understand the impact of risk and uncertainty in financial, project management, cost, and other forecasting models. It generates a range of possible outcomes and the probabilities they will occur for any choice of action.
        - **Risk-Return Scatter Plot**: A visual representation of the risk (standard deviation) and return (mean return) of each asset in the portfolio. This helps in understanding the risk-return profile of each asset.
        - **Sharpe Ratio Calculation**: Calculates the Sharpe ratio for the portfolio, providing a measure of the risk-adjusted return.

        The data is fetched using Yahoo Finance API and visualized using Plotly and Streamlit. This tool aims to help investors make informed decisions by providing key metrics and visualizations.

        **Developed by Vayk Mathrani**
    """)
    if st.button("Back to Dashboard") or back_button_top:
        st.experimental_rerun()
    st.stop()  # Stop further execution to only show the About page when button is clicked

# Input for stock tickers
tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT, AMZN):")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker]

# Main menu
menu_options = ["Portfolio Overview", "Stock Performance", "Portfolio Allocation", "Market Data", "Financial Ratios",
                "Portfolio Metrics"]
selected = st.selectbox("Select a tab", menu_options)

# Display selected tab
if selected == "Portfolio Overview":
    if tickers:
        portfolio_overview(tickers)
    else:
        st.write("Please enter stock tickers in the input field above to view the portfolio overview.")
elif selected == "Stock Performance":
    if tickers:
        stock_performance(tickers)
    else:
        st.write("Please enter stock tickers in the input field above to view stock performance.")
elif selected == "Portfolio Allocation":
    if tickers:
        app(tickers)
    else:
        st.write("Please enter stock tickers in the input field above to view portfolio allocation.")
elif selected == "Market Data":
    market_data()
elif selected == "Financial Ratios":
    if tickers:
        financial_ratios(tickers)
    else:
        st.write("Please enter stock tickers in the input field above to view financial ratios.")
elif selected == "Portfolio Metrics":
    if tickers:
        portfolio_metrics(tickers)
    else:
        st.write("Please enter stock tickers in the input field above to view portfolio metrics.")


# Function to fetch popular tickers from Yahoo Finance
def fetch_popular_tickers():
    url = "https://finance.yahoo.com/most-active"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tickers = []
    for a in soup.find_all('a', {'class': 'Fw(600) C($linkColor)'}):
        tickers.append(a.text)
    return tickers[:50]  # Limit to top 50 most active tickers


# Fetch company name
def fetch_company_name(ticker):
    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get('shortName', '')
    except Exception as e:
        st.error(f"Failed to fetch company name for: {ticker}")
        return ''


# Fetch stock data
def fetch_ticker_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            ticker_data = yf.Ticker(ticker).history(period="5d")
            if len(ticker_data) >= 5:
                current_price = ticker_data['Close'].iloc[-1]
                previous_price = ticker_data['Close'].iloc[-2]
                price_change = current_price - previous_price
                change_percent = (price_change / previous_price) * 100
                company_name = fetch_company_name(ticker)
                data[ticker] = {
                    "current_price": current_price,
                    "price_change": price_change,
                    "change_percent": change_percent,
                    "company_name": company_name
                }
        except Exception as e:
            st.error(f"Failed to fetch data for: {ticker}")
    return data


# Display stock ticker data on page load
popular_tickers = fetch_popular_tickers()
ticker_data = fetch_ticker_data(popular_tickers)

# Create HTML for the ticker
ticker_html = '<div id="ticker-container"><div id="ticker">'

# Add ticker items twice for seamless looping
for _ in range(2):  # Duplicate the list items
    for ticker, data in ticker_data.items():
        current_price = data["current_price"]
        price_change = data["price_change"]
        change_percent = data["change_percent"]
        company_name = data["company_name"]
        if price_change > 0:
            change_class = 'up'
            change_symbol = '▲'
        elif price_change < 0:
            change_class = 'down'
            change_symbol = '▼'
        else:
            change_class = 'no-change'
            change_symbol = '→'
        ticker_html += f'<div class="ticker-item"><div class="symbol-container"><div class="symbol">{ticker}</div><div class="company-name">{company_name}</div></div><div class="price-change"><span class="price">${current_price:.2f}</span> <span class="change {change_class}">{change_symbol} {price_change:.2f} ({change_percent:.2f}%)</span></div></div>'
ticker_html += '</div></div>'

# Display ticker in sidebar
st.sidebar.markdown(ticker_html, unsafe_allow_html=True)
