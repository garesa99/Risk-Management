import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.portfolio_analysis import calculate_portfolio_returns

st.set_page_config(layout="wide", page_title="Advanced Analysis")

st.title("📊 Advanced Analysis")

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    returns, weights = st.session_state.returns, st.session_state.weights

    portfolio_returns = calculate_portfolio_returns(returns, weights)

    st.subheader("Rolling Statistics")
    window = st.slider("Rolling Window (days)", 5, 252, 30)

    rolling_mean = portfolio_returns.rolling(window=window).mean()
    rolling_std = portfolio_returns.rolling(window=window).std()

    # Create an interactive chart with hover tooltips
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=portfolio_returns.index,
            y=portfolio_returns,
            mode='lines',
            name='Daily Returns',
            hovertemplate='Date: %{x}<br>Daily Return: %{y:.2%}<br><br>This line represents the daily returns of your portfolio.<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            mode='lines',
            name=f'{window}-day Rolling Mean',
            hovertemplate=f'Date: {{%x}}<br>{window}-day Rolling Mean: {{%y:.2%}}<br><br>This line shows the rolling average returns over the selected window.<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_std.index,
            y=rolling_std,
            mode='lines',
            name=f'{window}-day Rolling Std',
            hovertemplate=f'Date: {{%x}}<br>{window}-day Rolling Std Dev: {{%y:.2%}}<br><br>This line indicates the rolling standard deviation, a measure of volatility.<extra></extra>'
        )
    )
    fig.update_layout(
        title='Portfolio Returns with Rolling Statistics',
        xaxis_title='Date',
        yaxis_title='Returns',
        hovermode='x unified',
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the lines to see detailed explanations of each metric.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Matrix")
    correlation_matrix = returns.corr()

    # Create an interactive heatmap with hover tooltips
    heatmap = go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='Viridis',
        hovertemplate='Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.2f}<br><br>This value represents the correlation between the two assets.<extra></extra>'
    )
    fig_corr = go.Figure(data=heatmap)
    fig_corr.update_layout(
        title='Asset Correlation Matrix',
        xaxis_title='Assets',
        yaxis_title='Assets',
        margin=dict(l=60, r=60, t=60, b=60)
    )
    fig_corr.add_annotation(
        x=0.5,
        y=-0.1,
        xref='paper',
        yref='paper',
        text='Hover over the heatmap to see the correlation between assets.',
        showarrow=False
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Asset Performance Comparison")
    cumulative_returns = (1 + returns).cumprod()

    # Create an interactive line chart with hover tooltips
    fig_perf = go.Figure()
    for asset in cumulative_returns.columns:
        fig_perf.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[asset],
                mode='lines',
                name=asset,
                hovertemplate=f'Date: {{%x}}<br>{asset} Cumulative Return: {{%y:.2f}}x<br><br>This line represents the growth of ${1:.2f} invested in {asset} over time.<extra></extra>'
            )
        )
    fig_perf.update_layout(
        title='Cumulative Returns by Asset',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (Growth of $1)',
        hovermode='x unified',
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig_perf.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the lines to compare asset performance over time.',
        showarrow=False
    )
    st.plotly_chart(fig_perf, use_container_width=True)
