import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.portfolio_analysis import calculate_portfolio_returns
from src.risk_metrics import (
    calculate_var, calculate_cvar, calculate_beta,
    calculate_treynor_ratio, calculate_information_ratio, calculate_max_drawdown
)

st.set_page_config(layout="wide", page_title="Risk Metrics")

st.title("📉 Risk Metrics")

# Add custom CSS for tooltips
st.markdown("""
<style>
.metric-with-tooltip {
    position: relative;
    display: inline-block;
    margin: 10px;
    text-align: center;
}

.metric-with-tooltip .tooltiptext {
    visibility: hidden;
    width: 220px;
    background-color: #555;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position above */
    left: 50%;
    margin-left: -110px;
    opacity: 0;
    transition: opacity 0.3s;
}

.metric-with-tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

.metric-label {
    font-size: 16px;
    color: #666;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

if 'returns' not in st.session_state or 'weights' not in st.session_state:
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    returns, weights = st.session_state.returns, st.session_state.weights

    portfolio_returns = calculate_portfolio_returns(returns, weights)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Value at Risk (VaR)")
        confidence_level = st.slider("Confidence Level", 90, 99, 95)
        var = calculate_var(portfolio_returns, confidence_level / 100)
        st.markdown(f'''
        <div class="metric-with-tooltip">
          <div class="metric-label">VaR ({confidence_level}%)</div>
          <div class="metric-value">{var*100:.2f}%</div>
          <div class="tooltiptext">Value at Risk (VaR) estimates the maximum loss over a target horizon with a given confidence level.</div>
        </div>
        ''', unsafe_allow_html=True)

        st.subheader("Conditional Value at Risk (CVaR)")
        cvar = calculate_cvar(portfolio_returns, confidence_level / 100)
        st.markdown(f'''
        <div class="metric-with-tooltip">
          <div class="metric-label">CVaR ({confidence_level}%)</div>
          <div class="metric-value">{cvar*100:.2f}%</div>
          <div class="tooltiptext">Conditional VaR (CVaR) is the expected loss exceeding the VaR at the given confidence level.</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        st.subheader("Other Risk Metrics")
        # Assuming market_returns is provided; if not, replace with a benchmark return series
        market_returns = returns.mean(axis=1)
        risk_free_rate = 0.02 / 252  # Assuming a 2% annual risk-free rate

        beta = calculate_beta(portfolio_returns, market_returns)
        treynor_ratio = calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate)
        information_ratio = calculate_information_ratio(portfolio_returns, market_returns)
        max_drawdown = calculate_max_drawdown(portfolio_returns)

        metrics = [
            ("Beta", beta, "Beta measures the sensitivity of the portfolio's returns to market movements."),
            ("Treynor Ratio", treynor_ratio, "Treynor Ratio evaluates returns adjusted for systematic risk."),
            ("Information Ratio", information_ratio, "Information Ratio measures portfolio returns beyond the benchmark compared to the volatility of those returns."),
            ("Maximum Drawdown", max_drawdown * 100, "Maximum Drawdown represents the largest peak-to-trough decline in the portfolio's value.")
        ]

        for label, value, tooltip in metrics:
            st.markdown(f'''
            <div class="metric-with-tooltip">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value:.2f}{'%' if 'Drawdown' in label else ''}</div>
              <div class="tooltiptext">{tooltip}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.subheader("Returns Distribution")
    var_line = var
    cvar_line = cvar

    # Create an interactive histogram with hover tooltips
    hist = go.Histogram(
        x=portfolio_returns,
        nbinsx=50,
        name='Returns',
        hovertemplate='Return: %{x:.2%}<br>Frequency: %{y}<extra></extra>'
    )
    var_line_trace = go.Scatter(
        x=[var_line, var_line],
        y=[0, max(np.histogram(portfolio_returns, bins=50)[0])],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f'VaR ({confidence_level}%)',
        hovertemplate=f'VaR ({confidence_level}%): {var_line:.2%}<extra></extra>'
    )
    cvar_line_trace = go.Scatter(
        x=[cvar_line, cvar_line],
        y=[0, max(np.histogram(portfolio_returns, bins=50)[0])],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name=f'CVaR ({confidence_level}%)',
        hovertemplate=f'CVaR ({confidence_level}%): {cvar_line:.2%}<extra></extra>'
    )

    fig = go.Figure(data=[hist, var_line_trace, cvar_line_trace])
    fig.update_layout(
        title='Portfolio Returns Distribution',
        xaxis_title='Returns',
        yaxis_title='Frequency',
        hovermode='x unified',
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.add_annotation(
        x=0.5, y=-0.15, xref='paper', yref='paper',
        text='Hover over the histogram to see return frequencies and over the lines to see VaR and CVaR values.',
        showarrow=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Rolling Volatility")
    rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
    vol_trace = go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol,
        mode='lines',
        name='30-day Rolling Volatility',
        hovertemplate='Date: %{x}<br>Volatility: %{y:.2%}<extra></extra>'
    )
    fig_vol = go.Figure(data=[vol_trace])
    fig_vol.update_layout(
        title='30-day Rolling Annualized Volatility',
        xaxis_title='Date',
        yaxis_title='Volatility',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig_vol.add_annotation(
        x=0.5, y=-0.15, xref='paper', yref='paper',
        text='Hover over the line to see volatility on specific dates.',
        showarrow=False
    )

    st.plotly_chart(fig_vol, use_container_width=True)
