import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.portfolio_analysis import calculate_portfolio_returns
from src.risk_metrics import calculate_var, calculate_cvar

st.set_page_config(layout="wide", page_title="Stress Testing")

st.title("🔥 Stress Testing")

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
    width: 240px;
    background-color: #555;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position above */
    left: 50%;
    margin-left: -120px;
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

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    returns, weights = st.session_state.returns, st.session_state.weights

    portfolio_returns = calculate_portfolio_returns(returns, weights)

    st.subheader("Scenario Analysis")

    scenario = st.selectbox("Select Scenario", ["Market Crash", "Economic Recession", "Interest Rate Hike", "Custom"])

    if scenario == "Custom":
        shock_magnitude = st.slider("Shock Magnitude (%)", -50, 50, 0)
    else:
        shock_magnitudes = {
            "Market Crash": -30,
            "Economic Recession": -20,
            "Interest Rate Hike": -10
        }
        shock_magnitude = shock_magnitudes[scenario]

    shock_duration = st.slider("Shock Duration (days)", 1, 30, 5)

    # Apply shock to returns
    shocked_returns = portfolio_returns.copy()
    shock_per_day = (shock_magnitude / 100) / shock_duration
    shocked_returns.iloc[-shock_duration:] += shock_per_day

    # Calculate metrics
    original_cumulative_returns = (1 + portfolio_returns).cumprod()
    shocked_cumulative_returns = (1 + shocked_returns).cumprod()

    original_var = calculate_var(portfolio_returns, 0.95)
    shocked_var = calculate_var(shocked_returns, 0.95)

    original_cvar = calculate_cvar(portfolio_returns, 0.95)
    shocked_cvar = calculate_cvar(shocked_returns, 0.95)

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Original VaR (95%)</div>
            <div class="metric-value">{original_var*100:.2f}%</div>
            <div class="tooltiptext">
                Value at Risk (VaR) estimates the maximum potential loss over a given time frame at a specified confidence level under normal market conditions.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Original CVaR (95%)</div>
            <div class="metric-value">{original_cvar*100:.2f}%</div>
            <div class="tooltiptext">
                Conditional Value at Risk (CVaR) represents the average loss exceeding the VaR at the specified confidence level under normal conditions.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        var_diff = (shocked_var - original_var) * 100
        cvar_diff = (shocked_cvar - original_cvar) * 100
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Shocked VaR (95%)</div>
            <div class="metric-value">{shocked_var*100:.2f}%</div>
            <div class="tooltiptext">
                VaR after applying the stress scenario. This shows how the potential maximum loss changes under stressed conditions.
                <br><br>Change from original VaR: {var_diff:+.2f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Shocked CVaR (95%)</div>
            <div class="metric-value">{shocked_cvar*100:.2f}%</div>
            <div class="tooltiptext">
                CVaR after applying the stress scenario. This indicates the expected loss beyond the VaR under stressed conditions.
                <br><br>Change from original CVaR: {cvar_diff:+.2f}%
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=original_cumulative_returns.index,
        y=original_cumulative_returns,
        mode='lines',
        name='Original',
        hovertemplate='Date: %{x}<br>Original Cumulative Return: %{y:.2f}x<br><br>This line represents your portfolio\'s cumulative returns under normal conditions.<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=shocked_cumulative_returns.index,
        y=shocked_cumulative_returns,
        mode='lines',
        name='Shocked',
        hovertemplate='Date: %{x}<br>Shocked Cumulative Return: %{y:.2f}x<br><br>This line shows your portfolio\'s cumulative returns after applying the stress scenario.<extra></extra>'
    ))
    fig.update_layout(
        title='Original vs. Shocked Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (Growth of $1)',
        hovermode='x unified',
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the lines to compare portfolio performance under normal and stressed conditions.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)
