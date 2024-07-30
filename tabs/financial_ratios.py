import streamlit as st
import pandas as pd
import plotly.express as px
from tabs.portfolio_overview import get_stock_data


def financial_ratios(tickers):
    data, failed_tickers = get_stock_data(tickers)  # Unpack the returned tuple
    if data.empty:
        st.error("No valid data available for the selected tickers.")
        return

    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    corr_matrix = returns.corr()

    st.subheader("Financial Ratios")

    # Mean Returns
    st.markdown("### Mean Returns")
    mean_returns_df = mean_returns.reset_index()
    mean_returns_df.columns = ['Ticker', 'Mean Return']
    fig_mean_returns = px.bar(mean_returns_df, x='Ticker', y='Mean Return', title='Mean Returns')
    st.plotly_chart(fig_mean_returns)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Covariance Matrix
    st.markdown("### Covariance Matrix")
    fig_cov = px.imshow(cov_matrix, text_auto=True, title='Covariance Matrix', labels={'color': 'Covariance'})
    st.plotly_chart(fig_cov)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Correlation Matrix
    st.markdown("### Correlation Matrix")
    fig_corr = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix', labels={'color': 'Correlation'})
    st.plotly_chart(fig_corr)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Beta Calculation
    st.subheader("Beta Calculation")

    market_indexes_input = st.text_input("Enter market index tickers for Beta calculation (comma separated, e.g., ^GSPC, ^DJI):")
    if market_indexes_input:
        market_indexes = [index.strip() for index in market_indexes_input.split(',')]
        betas = {}
        for market_index in market_indexes:
            try:
                market_data, _ = get_stock_data([market_index])  # Using user-provided market index

                if isinstance(market_data, pd.Series):
                    market_data = market_data.to_frame(name=market_index)

                if market_index in market_data.columns:
                    market_returns = market_data.pct_change().dropna()
                    for ticker in tickers:
                        if ticker in returns.columns:
                            cov = returns[ticker].cov(market_returns[market_index])
                            var = market_returns[market_index].var()
                            beta = cov / var
                            betas[f"{ticker} vs {market_index}"] = beta
                else:
                    st.error(f"Market data for {market_index} not available.")
            except Exception as e:
                st.error(f"Error fetching market data for {market_index}: {e}")

        if betas:
            beta_df = pd.DataFrame(list(betas.items()), columns=['Ticker vs Market Index', 'Beta'])

            # Plot Beta Values
            fig_beta = px.bar(beta_df, x='Ticker vs Market Index', y='Beta', title='Beta Values')
            st.plotly_chart(fig_beta)
        else:
            st.error("No beta values calculated.")
    else:
        st.write("Please enter market index tickers for Beta calculation.")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Sharpe Ratio Calculation
    st.subheader("Sharpe Ratio Calculation")
    risk_free_rate = 0.01  # Assume a risk-free rate of 1%
    sharpe_ratios = (mean_returns - risk_free_rate) / returns.std()
    sharpe_ratios_df = sharpe_ratios.reset_index()
    sharpe_ratios_df.columns = ['Ticker', 'Sharpe Ratio']
    fig_sharpe = px.bar(sharpe_ratios_df, x='Ticker', y='Sharpe Ratio', title='Sharpe Ratios')
    st.plotly_chart(fig_sharpe)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Sortino Ratio
    st.subheader("Sortino Ratio")
    downside_returns = returns[returns < 0]
    sortino_ratios = (returns.mean() - risk_free_rate) / downside_returns.std()
    sortino_ratios_df = sortino_ratios.reset_index()
    sortino_ratios_df.columns = ['Ticker', 'Sortino Ratio']
    fig_sortino = px.bar(sortino_ratios_df, x='Ticker', y='Sortino Ratio', title='Sortino Ratios')
    st.plotly_chart(fig_sortino)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Maximum Drawdown Calculation
    st.subheader("Maximum Drawdown")
    cumulative_returns = (1 + returns).cumprod()
    cumulative_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - cumulative_max) / cumulative_max
    max_drawdowns = drawdowns.min().reset_index()
    max_drawdowns.columns = ['Ticker', 'Maximum Drawdown']
    fig_drawdowns = px.bar(max_drawdowns, x='Ticker', y='Maximum Drawdown', title='Maximum Drawdowns')
    st.plotly_chart(fig_drawdowns)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Volatility Calculation
    st.subheader("Volatility")
    volatilities = returns.std().reset_index()
    volatilities.columns = ['Ticker', 'Volatility']
    fig_volatility = px.bar(volatilities, x='Ticker', y='Volatility', title='Volatility')
    st.plotly_chart(fig_volatility)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)