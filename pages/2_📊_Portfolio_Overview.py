import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.portfolio_analysis import calculate_portfolio_returns, calculate_metrics

st.set_page_config(layout="wide", page_title="Portfolio Overview")

st.title("📊 Portfolio Overview")

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    df, returns, weights, portfolio_value = st.session_state.df, st.session_state.returns, st.session_state.weights, st.session_state.portfolio_value
    
    # Display current portfolio composition
    st.subheader("Current Portfolio Composition")
    composition_df = pd.DataFrame({
        'Asset': weights.index,
        'Weight': weights,
        'Current Price': df.iloc[-1],
        'Current Value': df.iloc[-1] * weights * portfolio_value
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(labels=composition_df['Asset'], values=composition_df['Weight'], hole=.3)])
        fig.update_layout(title='Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(composition_df.style.format({'Weight': '{:.2%}', 'Current Price': '${:.2f}', 'Current Value': '${:.2f}'}))
    
    # Display key metrics
    st.subheader("Key Performance Metrics")
    portfolio_returns = calculate_portfolio_returns(returns, weights)
    metrics = calculate_metrics(portfolio_returns, portfolio_value)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Portfolio Value", f"${metrics['portfolio_value']:,.2f}")
        st.metric("Total Return", f"{metrics['total_return']:.2f}%")
    with col2:
        st.metric("Daily Return", f"{metrics['daily_return']:.2f}%")
        st.metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
    with col4:
        st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    
    # Portfolio Performance Chart
    st.subheader("Portfolio Performance")
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Portfolio'))
    fig.update_layout(title='Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')
    st.plotly_chart(fig, use_container_width=True)

    # Asset Performance Comparison
    st.subheader("Asset Performance Comparison")
    asset_returns = (1 + returns).cumprod()
    fig = go.Figure()
    for asset in asset_returns.columns:
        fig.add_trace(go.Scatter(x=asset_returns.index, y=asset_returns[asset], mode='lines', name=asset))
    fig.update_layout(title='Asset Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')
    st.plotly_chart(fig, use_container_width=True)