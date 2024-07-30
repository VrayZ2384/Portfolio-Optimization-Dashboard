import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from tabs.portfolio_overview import get_stock_data
import numpy as np
import yfinance as yf

def portfolio_metrics(tickers):
    data, failed_tickers = get_stock_data(tickers)  # Unpack the returned tuple
    if data.empty:
        st.error("No valid data available for the selected tickers.")
        return

    returns = data.pct_change().dropna()
    cumulative_returns = (1 + returns).cumprod() - 1

    st.subheader("Portfolio Metrics")

    # Cumulative Returns
    st.markdown("### Cumulative Returns")
    fig_cum_returns = go.Figure()
    for ticker in cumulative_returns.columns:
        fig_cum_returns.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns[ticker],
            mode='lines',
            name=ticker,
            hovertemplate=f"{ticker} Cumulative Return: %{{y:.2%}}<extra></extra>"
        ))
    fig_cum_returns.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return', hovermode="x unified")
    st.plotly_chart(fig_cum_returns)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Volatility
    st.markdown("### Volatility (Standard Deviation)")
    vol_df = returns.std().reset_index()
    vol_df.columns = ['Ticker', 'Volatility']
    fig_vol = px.bar(vol_df, x='Ticker', y='Volatility', title='Volatility (Standard Deviation)')
    st.plotly_chart(fig_vol)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Sharpe Ratio
    st.subheader("Sharpe Ratio")
    risk_free_rate = 0.01  # Assume a risk-free rate of 1%
    sharpe_ratios = (returns.mean() - risk_free_rate) / returns.std()
    sharpe_ratios_df = sharpe_ratios.reset_index()
    sharpe_ratios_df.columns = ['Ticker', 'Sharpe Ratio']
    fig_sharpe = px.bar(sharpe_ratios_df, x='Ticker', y='Sharpe Ratio', title='Sharpe Ratios')
    st.plotly_chart(fig_sharpe)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    fig_corr = px.imshow(returns.corr(), text_auto=True, title='Correlation Matrix', labels={'color': 'Correlation'})
    st.plotly_chart(fig_corr)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # VaR and CVaR
    st.subheader("Value at Risk (VaR) and Conditional VaR (CVaR)")
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    var_df = pd.DataFrame({'Ticker': var_95.index, 'VaR (95%)': var_95.values, 'CVaR (95%)': cvar_95.values})
    fig_var = px.bar(var_df, x='Ticker', y=['VaR (95%)', 'CVaR (95%)'], barmode='group', title='VaR and CVaR (95% confidence level)')
    st.plotly_chart(fig_var)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Rolling Metrics
    st.subheader("Rolling Metrics")
    rolling_window = st.slider("Select rolling window size (days):", 20, 252, 60)
    rolling_mean = returns.rolling(window=rolling_window).mean()
    rolling_vol = returns.rolling(window=rolling_window).std()

    st.markdown("### Rolling Mean")
    fig_rolling_mean = go.Figure()
    for ticker in rolling_mean.columns:
        fig_rolling_mean.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean[ticker],
            mode='lines',
            name=ticker,
            hovertemplate=f"{ticker} Rolling Mean: %{{y:.2%}}<extra></extra>"
        ))
    fig_rolling_mean.update_layout(title='Rolling Mean', xaxis_title='Date', yaxis_title='Rolling Mean', hovermode="x unified")
    st.plotly_chart(fig_rolling_mean)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("### Rolling Volatility")
    fig_rolling_vol = go.Figure()
    for ticker in rolling_vol.columns:
        fig_rolling_vol.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[ticker],
            mode='lines',
            name=ticker,
            hovertemplate=f"{ticker} Rolling Volatility: %{{y:.2%}}<extra></extra>"
        ))
    fig_rolling_vol.update_layout(title='Rolling Volatility', xaxis_title='Date', yaxis_title='Rolling Volatility', hovermode="x unified")
    st.plotly_chart(fig_rolling_vol)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Dividend Yield
    st.subheader("Dividend Yield")
    dividend_yield = {}
    no_dividend_stocks = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'dividendYield' in info and info['dividendYield'] is not None:
            dividend_yield[ticker] = info['dividendYield'] * 100
        else:
            dividend_yield[ticker] = 0
            no_dividend_stocks.append(ticker)

    dividend_yield_df = pd.DataFrame(list(dividend_yield.items()), columns=['Ticker', 'Dividend Yield (%)'])
    fig_dividend_yield = px.bar(dividend_yield_df, x='Ticker', y='Dividend Yield (%)', title='Dividend Yield (%)')
    st.plotly_chart(fig_dividend_yield)

    if no_dividend_stocks:
        st.write(f"The following stocks do not have a dividend yield: {', '.join(no_dividend_stocks)}")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Risk-Return Scatter Plot
    st.subheader("Risk-Return Scatter Plot")
    risk_return_df = pd.DataFrame({
        'Ticker': returns.columns,
        'Risk (Std Dev)': returns.std().values,
        'Return': returns.mean().values
    })
    fig_risk_return = px.scatter(risk_return_df, x='Risk (Std Dev)', y='Return', text='Ticker', title='Risk vs Return', width=800, height=600)
    fig_risk_return.update_traces(textposition='top center')
    st.plotly_chart(fig_risk_return)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Monte Carlo Simulation
    st.subheader("Monte Carlo Simulation")
    num_simulations = st.slider("Number of Simulations:", 100, 10000, 1000)
    time_horizon = st.slider("Time Horizon (days):", 30, 252, 100)

    # Initialize session state for Monte Carlo simulation
    if 'simulations' not in st.session_state:
        st.session_state.simulations = None
        st.session_state.simulations_completed = 0

    # Run simulations if not already done
    if st.session_state.simulations is None or st.session_state.simulations_completed < num_simulations:
        progress_bar = st.progress(st.session_state.simulations_completed / num_simulations)
        simulations = np.zeros((time_horizon + 1, num_simulations))  # +1 to include the initial value
        last_prices = data.iloc[-1]

        for i in range(st.session_state.simulations_completed, num_simulations):
            simulation_df = pd.DataFrame()
            for ticker in tickers:
                price_series = [last_prices[ticker]]
                for j in range(time_horizon):
                    price = price_series[-1] * (1 + np.random.normal(returns[ticker].mean(), returns[ticker].std()))
                    price_series.append(price)
                simulation_df[ticker] = price_series
            simulations[:, i] = simulation_df.sum(axis=1)
            st.session_state.simulations_completed += 1
            progress_bar.progress(st.session_state.simulations_completed / num_simulations)

        st.session_state.simulations = simulations

    # Plot Monte Carlo simulation
    fig_monte_carlo = go.Figure()
    for i in range(num_simulations):
        fig_monte_carlo.add_trace(go.Scatter(
            x=list(range(time_horizon + 1)),
            y=st.session_state.simulations[:, i],
            mode='lines',
            opacity=0.1,
            line=dict(color='blue'),
            showlegend=False
        ))

    fig_monte_carlo.update_layout(title='Monte Carlo Simulation', xaxis_title='Days', yaxis_title='Portfolio Value')
    st.plotly_chart(fig_monte_carlo)