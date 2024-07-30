import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from newsapi import NewsApiClient
from tabs.portfolio_overview import get_stock_data
import time


# Function to get sector performance
def get_sector_performance():
    sectors = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Finance': 'XLF',
        'Energy': 'XLE',
        'Consumer Goods': 'XLP'
    }
    sector_performance = {}
    for sector, ticker in sectors.items():
        try:
            data = yf.download(ticker, period="1mo")['Adj Close']
            performance = (data[-1] - data[0]) / data[0]
            sector_performance[sector] = performance
        except Exception as e:
            st.error(f"Failed to download data for sector: {sector} - {str(e)}")
    df = pd.DataFrame(list(sector_performance.items()), columns=['sector', 'performance'])
    return df


# Function to get financial news
def get_financial_news(api_key, query):
    newsapi = NewsApiClient(api_key=api_key)
    today = datetime.today().strftime('%Y-%m-%d')
    one_month_ago = (datetime.today() - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    retries = 3

    for attempt in range(retries):
        try:
            everything = newsapi.get_everything(q=query, language='en', from_param=one_month_ago, to=today,
                                                sort_by='publishedAt', page_size=20)
            articles = everything['articles']
            valid_articles = [article for article in articles if
                              '[Removed]' not in article['title'] and '[Removed]' not in article['description']]
            return valid_articles
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                st.error(f"Unauthorized access - please check your API key. Attempt {attempt + 1} of {retries}.")
            else:
                st.error(f"Failed to fetch news articles: {str(e)}. Attempt {attempt + 1} of {retries}.")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                st.error("Max retries reached. Could not fetch news articles.")
    return []


def get_stock_data(tickers, start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    data = {}
    failed_tickers = []

    for ticker in tickers:
        attempt = 0
        success = False
        while attempt < 3 and not success:
            try:
                ticker_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
                if ticker_data.empty:
                    raise Exception("No data found")
                data[ticker] = ticker_data
                success = True
            except Exception as e:
                attempt += 1
                if attempt < 3:
                    time.sleep(2)  # Wait before retrying
                else:
                    failed_tickers.append(ticker)
                    st.error(f"Failed to download data for: {ticker} after 3 attempts")

    return pd.DataFrame(data), failed_tickers


def market_data():
    st.markdown("## Market Data")

    # Get user inputs for indices
    indices_input = st.text_input("Enter market indices separated by commas (e.g., ^GSPC, ^DJI, ^IXIC, ^RUT, ^VIX):")
    indices = [ticker.strip().upper() for ticker in indices_input.split(',') if ticker]

    if indices:
        # Get stock data
        data, failed_tickers = get_stock_data(indices)

        # Display error message for failed tickers
        if failed_tickers:
            st.error(f"Failed to download data for: {', '.join(failed_tickers)}")

        if not data.empty:
            # Descriptive Statistics Table
            st.markdown("### Descriptive Statistics")
            st.write(data.describe())

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Plot Market Data
            st.markdown("### Market Indices")
            fig = go.Figure()
            for ticker in indices:
                fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))
            fig.update_layout(title='Market Data', xaxis_title='Date', yaxis_title='Price', hovermode="x unified")
            st.plotly_chart(fig)

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Daily returns
            st.markdown("### Daily Returns")
            daily_returns = data.pct_change().dropna()
            fig = px.line(daily_returns, title='Daily Returns')
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig)

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Cumulative returns
            st.markdown("### Cumulative Returns")
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            fig = go.Figure()
            for ticker in cumulative_returns.columns:
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns[ticker],
                    mode='lines',
                    name=ticker,
                    hovertemplate=f"{ticker} Cumulative Return: %{{y:.2%}}<extra></extra>"
                ))
            fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return',
                              hovermode="x unified")
            st.plotly_chart(fig)

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Performance metrics
            st.markdown("### Performance Metrics")
            annualized_return = daily_returns.mean() * 252
            annualized_volatility = daily_returns.std() * (252 ** 0.5)
            sharpe_ratio = annualized_return / annualized_volatility

            st.markdown(
                f"**Annualized Return:** <span class='hover-info' title='Annualized Return = Daily Return Mean * 252'>{annualized_return.mean():.2%}</span>",
                unsafe_allow_html=True)
            st.markdown(
                f"**Annualized Volatility:** <span class='hover-info' title='Annualized Volatility = Daily Return Std Dev * sqrt(252)'>{annualized_volatility.mean():.2%}</span>",
                unsafe_allow_html=True)
            st.markdown(
                f"**Sharpe Ratio:** <span class='hover-info' title='Sharpe Ratio = Annualized Return / Annualized Volatility'>{sharpe_ratio.mean():.2f}</span>",
                unsafe_allow_html=True)

            # Max Drawdown
            cumulative_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - cumulative_max) / cumulative_max
            max_drawdown = drawdown.min()

            st.markdown(
                f"**Max Drawdown:** <span class='hover-info' title='Max Drawdown = Minimum Drawdown from Peak to Trough'>{max_drawdown.min():.2%}</span>",
                unsafe_allow_html=True)

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Visualize drawdown
            st.markdown("### Drawdown")
            fig = go.Figure()
            for ticker in drawdown.columns:
                fig.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown[ticker],
                    mode='lines',
                    name=ticker,
                    hovertemplate=f"{ticker} Drawdown: %{{y:.2%}}<extra></extra>"
                ))
            fig.update_layout(title='Drawdown', xaxis_title='Date', yaxis_title='Drawdown', hovermode="x unified")
            st.plotly_chart(fig)

            # Horizontal barrier
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)

            # Correlation matrix
            st.markdown("### Correlation Matrix")
            corr_matrix = daily_returns.corr()
            fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix')
            st.plotly_chart(fig)

    # Financial News
    st.markdown("### Financial News")
    news_api_key = st.secrets["general"]["news_api_key"]
    news_query = st.text_input("Enter a query for financial news (e.g., stock market, NASDAQ, tesla, etc.):")
    if news_query:
        try:
            articles = get_financial_news(news_api_key, news_query)
            if not articles:
                st.write("No news articles found for the query.")
            with st.expander("Click to view news articles"):
                for article in articles:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.write(article['description'])
                    st.write("---")  # Divider between articles
        except Exception as e:
            st.error(f"Failed to fetch news articles: {str(e)}")

    # Sector and Industry Analysis
    st.markdown("### Sector and Industry Analysis")
    sector_performance = get_sector_performance()
    fig = px.bar(sector_performance, x='sector', y='performance', title='Sector Performance')
    st.plotly_chart(fig)
