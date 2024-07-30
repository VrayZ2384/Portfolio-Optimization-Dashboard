import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


def get_stock_data(tickers, start_date="2020-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    data = {}
    failed_tickers = []
    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            if ticker_data.empty:
                raise Exception("No data found")
            data[ticker] = ticker_data
        except Exception as e:
            failed_tickers.append(ticker)
            st.error(f"Failed to download data for: {ticker}")
    return pd.DataFrame(data), failed_tickers


def portfolio_overview(tickers):
    data, failed_tickers = get_stock_data(tickers)
    st.markdown("## Portfolio Overview")

    # Display error message for failed tickers
    if failed_tickers:
        st.error(f"Failed to download data for: {', '.join(failed_tickers)}")

    if not data.empty:
        # Filter out columns with all zeros
        data = data.loc[:, (data != 0).any(axis=0)]

        # Descriptive Statistics Table
        st.markdown("### Descriptive Statistics")
        st.write(data.describe())

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Total portfolio value
        # Ensure total value calculation handles missing data correctly
        total_value = data.iloc[-1].sum()
        st.markdown(f"### Total Portfolio Value: **${total_value:,.2f}**")

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Daily returns
        daily_returns = data.pct_change().dropna()
        st.markdown("### Daily Returns")
        fig = px.line(daily_returns, title='Daily Returns')
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig)

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        st.markdown("### Cumulative Returns")
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

        # Portfolio performance metrics
        annualized_return = daily_returns.mean() * 252
        annualized_volatility = daily_returns.std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / annualized_volatility

        st.markdown("### Performance Metrics")
        st.markdown(
            f"**Annualized Return:** <span class='hover-info' title='Annualized Return = Daily Return Mean * 252'>{annualized_return.mean():.2%}</span>",
            unsafe_allow_html=True)
        st.markdown(
            f"**Annualized Volatility:** <span class='hover-info' title='Annualized Volatility = Daily Return Std Dev * sqrt(252)'>{annualized_volatility.mean():.2%}</span>",
            unsafe_allow_html=True)
        st.markdown(
            f"**Sharpe Ratio:** <span class='hover-info' title='Sharpe Ratio = Annualized Return / Annualized Volatility'>{sharpe_ratio.mean():.2f}</span>",
            unsafe_allow_html=True)

        # Additional performance metrics: Max Drawdown
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
    else:
        st.write("No valid data available for the selected tickers.")


