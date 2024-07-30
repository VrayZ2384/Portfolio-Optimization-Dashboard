import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from tabs.portfolio_overview import get_stock_data

def stock_performance(tickers):
    data, failed_tickers = get_stock_data(tickers)
    st.markdown("## Stock Performance Overview")

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
        total_value = data.iloc[-1].sum()
        st.markdown(f"### Total Portfolio Value: **${total_value:,.2f}**")

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Stock performance plot
        fig = go.Figure()
        for ticker in tickers:
            if ticker in data.columns:
                fig.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=ticker))

        fig.update_layout(title='Stock Performance', xaxis_title='Date', yaxis_title='Price', hovermode="x unified")
        st.plotly_chart(fig)

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Moving averages
        st.subheader("Moving Averages")
        for ticker in tickers:
            if ticker in data.columns:
                data[f"{ticker}_MA50"] = data[ticker].rolling(window=50).mean()
                data[f"{ticker}_MA200"] = data[ticker].rolling(window=200).mean()
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=f"{ticker} Price"))
                fig_ma.add_trace(go.Scatter(x=data.index, y=data[f"{ticker}_MA50"], mode='lines', name=f"{ticker} 50MA"))
                fig_ma.add_trace(go.Scatter(x=data.index, y=data[f"{ticker}_MA200"], mode='lines', name=f"{ticker} 200MA"))
                fig_ma.update_layout(title=f'{ticker} Moving Averages', xaxis_title='Date', yaxis_title='Price', hovermode="x unified")
                st.plotly_chart(fig_ma)

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Bollinger Bands
        st.subheader("Bollinger Bands")
        for ticker in tickers:
            if ticker in data.columns:
                data[f"{ticker}_20MA"] = data[ticker].rolling(window=20).mean()
                data[f"{ticker}_20STD"] = data[ticker].rolling(window=20).std()
                data[f"{ticker}_UpperBB"] = data[f"{ticker}_20MA"] + (data[f"{ticker}_20STD"] * 2)
                data[f"{ticker}_LowerBB"] = data[f"{ticker}_20MA"] - (data[f"{ticker}_20STD"] * 2)
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=data.index, y=data[ticker], mode='lines', name=f"{ticker} Price"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data[f"{ticker}_UpperBB"], mode='lines', name=f"{ticker} Upper BB"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data[f"{ticker}_LowerBB"], mode='lines', name=f"{ticker} Lower BB"))
                fig_bb.update_layout(title=f'{ticker} Bollinger Bands', xaxis_title='Date', yaxis_title='Price', hovermode="x unified")
                st.plotly_chart(fig_bb)

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    else:
        st.write("No valid data available for the selected tickers.")