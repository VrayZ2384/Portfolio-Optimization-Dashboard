import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import norm
from tabs.portfolio_overview import get_stock_data

def mean_variance_optimization(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return returns, std, returns / std

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
        return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


def risk_parity_optimization(data):
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()

    num_assets = len(cov_matrix)
    args = (cov_matrix,)

    def risk_budget_objective(x, cov_matrix):
        portfolio_var = np.dot(x.T, np.dot(cov_matrix, x))
        marginal_risk_contribs = np.dot(cov_matrix, x)
        risk_contribs = x * marginal_risk_contribs
        total_risk_contrib = np.sum(risk_contribs)
        risk_target = total_risk_contrib / num_assets
        return np.sum((risk_contribs - risk_target) ** 2)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(risk_budget_objective, num_assets * [1. / num_assets, ], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


def max_diversification_optimization(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def diversification_ratio(weights, mean_returns, cov_matrix):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        weighted_volatilities = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
        return -weighted_volatilities / portfolio_volatility

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(diversification_ratio, num_assets * [1. / num_assets, ], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


def cvar_optimization(data):
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    num_assets = len(mean_returns)
    alpha = 0.95
    args = (mean_returns, cov_matrix, alpha)

    def portfolio_performance(weights, mean_returns, cov_matrix, alpha):
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        var = norm.ppf(1 - alpha) * portfolio_volatility
        cvar = portfolio_return - var
        return portfolio_return, portfolio_volatility, cvar

    def neg_cvar(weights, mean_returns, cov_matrix, alpha):
        return -portfolio_performance(weights, mean_returns, cov_matrix, alpha)[2]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(neg_cvar, num_assets * [1. / num_assets, ], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x


def plot_portfolio_allocation(title, weights, labels, total_investment=None):
    formatted_weights = [f"{w * 100:.2f}%" for w in weights]
    fig = go.Figure(data=[go.Pie(labels=labels, values=weights, text=formatted_weights, textinfo='label+text')])
    fig.update_layout(title=title)
    st.plotly_chart(fig)

    if total_investment is not None:
        allocations = allocate_investment(total_investment, weights, labels)
        st.write(f"Investment Distribution for {title}:")
        for label, amount in allocations.items():
            st.write(f"{label}: ${amount:.2f}")


def plot_efficient_frontier(mean_returns, cov_matrix):
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weight_array = []

    np.random.seed(42)
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weight_array.append(weights)
        portfolio_return = np.sum(weights * mean_returns) * 252
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        results[0, i] = portfolio_stddev
        results[1, i] = portfolio_return
        results[2, i] = results[1, i] / results[0, i]  # Sharpe ratio

    max_sharpe_idx = np.argmax(results[2])
    fig = go.Figure(
        data=[
            go.Scatter(
                x=results[0],
                y=results[1],
                mode='markers',
                marker=dict(
                    color=results[2],
                    colorscale='Viridis',
                    size=5,
                    colorbar=dict(title='Sharpe Ratio', x=1.1)
                ),
                name='Portfolios'
            ),
            go.Scatter(
                x=[results[0, max_sharpe_idx]],
                y=[results[1, max_sharpe_idx]],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Max Sharpe Ratio'
            )
        ],
        layout=go.Layout(
            title='Efficient Frontier',
            xaxis=dict(title='Risk (Standard Deviation)'),
            yaxis=dict(title='Return'),
            legend=dict(orientation='h', y=-0.3),
            showlegend=True
        )
    )
    st.plotly_chart(fig)


def allocate_investment(total_investment, weights, labels):
    allocations = {label: weight * total_investment for label, weight in zip(labels, weights)}
    return allocations


def portfolio_recommendations(tickers):
    data, failed_tickers = get_stock_data(tickers)
    st.markdown("## Overall Portfolio Recommendations")

    if failed_tickers:
        st.error(f"Failed to download data for: {', '.join(failed_tickers)}")

    if not data.empty:
        data = data.loc[:, (data != 0).any(axis=0)]
        labels = data.columns

        # Calculate weights using different optimization techniques
        mv_weights = mean_variance_optimization(data)
        rp_weights = risk_parity_optimization(data)
        md_weights = max_diversification_optimization(data)
        cvar_weights = cvar_optimization(data)

        # Define different types of investment mindsets
        investment_mindsets = {
            "Conservative": {
                "description": "Low risk tolerance, focus on preserving capital.",
                "weights": (mv_weights * 0.1 + rp_weights * 0.7 + md_weights * 0.1 + cvar_weights * 0.1)
            },
            "Balanced": {
                "description": "Moderate risk tolerance, balanced focus on growth and stability.",
                "weights": (mv_weights * 0.3 + rp_weights * 0.3 + md_weights * 0.2 + cvar_weights * 0.2)
            },
            "Growth": {
                "description": "Higher risk tolerance, focus on capital appreciation.",
                "weights": (mv_weights * 0.4 + rp_weights * 0.1 + md_weights * 0.3 + cvar_weights * 0.2)
            },
            "Aggressive": {
                "description": "High risk tolerance, seeking maximum returns with high risk.",
                "weights": (mv_weights * 0.5 + rp_weights * 0.1 + md_weights * 0.2 + cvar_weights * 0.2)
            }
        }

        # Display the recommendations in a 2x2 grid layout
        st.markdown("### Based on Various Investment Mindsets")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Conservative")
            st.write(investment_mindsets["Conservative"]["description"])
            conservative_investment = st.number_input("Enter your investment amount for Conservative", min_value=0.0,
                                                      step=1000.0)
            plot_portfolio_allocation('Conservative Allocation', investment_mindsets["Conservative"]["weights"], labels,
                                      total_investment=conservative_investment)

        with col2:
            st.markdown("#### Balanced")
            st.write(investment_mindsets["Balanced"]["description"])
            balanced_investment = st.number_input("Enter your investment amount for Balanced", min_value=0.0,
                                                  step=1000.0)
            plot_portfolio_allocation('Balanced Allocation', investment_mindsets["Balanced"]["weights"], labels,
                                      total_investment=balanced_investment)

        with col1:
            st.markdown("#### Growth")
            st.write(investment_mindsets["Growth"]["description"])
            growth_investment = st.number_input("Enter your investment amount for Growth", min_value=0.0, step=1000.0)
            plot_portfolio_allocation('Growth Allocation', investment_mindsets["Growth"]["weights"], labels,
                                      total_investment=growth_investment)

        with col2:
            st.markdown("#### Aggressive")
            st.write(investment_mindsets["Aggressive"]["description"])
            aggressive_investment = st.number_input("Enter your investment amount for Aggressive", min_value=0.0,
                                                    step=1000.0)
            plot_portfolio_allocation('Aggressive Allocation', investment_mindsets["Aggressive"]["weights"], labels,
                                      total_investment=aggressive_investment)
    else:
        st.write("No valid data available for the selected tickers.")


def portfolio_allocation(tickers):
    data, failed_tickers = get_stock_data(tickers)
    st.markdown("## Portfolio Allocation")

    if failed_tickers:
        st.error(f"Failed to download data for: {', '.join(failed_tickers)}")

    if not data.empty:
        data = data.loc[:, (data != 0).any(axis=0)]
        labels = data.columns

        # Mean-Variance Optimization
        mv_weights = mean_variance_optimization(data)
        st.markdown("### Mean-Variance Optimization")
        mv_investment = st.number_input("Enter your investment amount for Mean-Variance", min_value=0.0, step=1000.0)
        plot_portfolio_allocation('Mean-Variance Optimization', mv_weights, labels, total_investment=mv_investment)

        # Risk Parity Optimization
        rp_weights = risk_parity_optimization(data)
        st.markdown("### Risk Parity Optimization")
        rp_investment = st.number_input("Enter your investment amount for Risk Parity", min_value=0.0, step=1000.0)
        plot_portfolio_allocation('Risk Parity Optimization', rp_weights, labels, total_investment=rp_investment)

        # Maximum Diversification Optimization
        md_weights = max_diversification_optimization(data)
        st.markdown("### Maximum Diversification Optimization")
        md_investment = st.number_input("Enter your investment amount for Maximum Diversification", min_value=0.0,
                                        step=1000.0)
        plot_portfolio_allocation('Maximum Diversification Optimization', md_weights, labels,
                                  total_investment=md_investment)

        # Mean-CVaR Optimization
        cvar_weights = cvar_optimization(data)
        st.markdown("### Mean-CVaR Optimization")
        cvar_investment = st.number_input("Enter your investment amount for Mean-CVaR", min_value=0.0, step=1000.0)
        plot_portfolio_allocation('Mean-CVaR Optimization', cvar_weights, labels, total_investment=cvar_investment)

        # Plot efficient frontier for Mean-Variance Optimization
        st.markdown("### Efficient Frontier (Mean-Variance Optimization)")
        returns = data.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        plot_efficient_frontier(mean_returns, cov_matrix)

        # Cumulative returns
        cumulative_returns = (data / data.iloc[0]) - 1
        st.markdown("### Cumulative Returns")
        fig = go.Figure()
        for ticker in data.columns:
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns[ticker], mode='lines', name=ticker))
        fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return',
                          hovermode="x unified")
        st.plotly_chart(fig)

        # Daily returns
        daily_returns = data.pct_change().dropna()
        st.markdown("### Daily Returns")
        fig = go.Figure()
        for ticker in data.columns:
            fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns[ticker], mode='lines', name=ticker))
        fig.update_layout(title='Daily Returns', xaxis_title='Date', yaxis_title='Daily Return', hovermode="x unified")
        st.plotly_chart(fig)

        # Horizontal barrier
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    else:
        st.write("No valid data available for the selected tickers.")


# Include portfolio recommendations
def app(tickers):
    portfolio_allocation(tickers)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    portfolio_recommendations(tickers)
