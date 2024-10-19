import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(layout="wide", page_title="Portfolio Optimization")

st.title("📈 Portfolio Optimization")

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
    returns = st.session_state.returns
    weights = st.session_state.weights

    # Utility functions for optimization
    def portfolio_return(weights, returns):
        return np.sum(returns.mean() * weights) * 252

    def portfolio_volatility(weights, returns):
        return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    def sharpe_ratio(weights, returns, risk_free_rate):
        p_ret = portfolio_return(weights, returns)
        p_vol = portfolio_volatility(weights, returns)
        return (p_ret - risk_free_rate) / p_vol

    # Optimization
    st.subheader("Portfolio Optimization")

    optimization_goal = st.selectbox("Optimization Goal", ["Maximize Sharpe Ratio", "Minimize Volatility"])
    risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 5.0, 2.0) / 100

    num_assets = len(returns.columns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    if optimization_goal == "Maximize Sharpe Ratio":
        result = minimize(lambda w: -sharpe_ratio(w, returns, risk_free_rate),
                          weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(lambda w: portfolio_volatility(w, returns),
                          weights, method='SLSQP', bounds=bounds, constraints=constraints)

    optimized_weights = result.x

    # Display results
    st.subheader("Optimization Results")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Optimized Return</div>
            <div class="metric-value">{portfolio_return(optimized_weights, returns):.2%}</div>
            <div class="tooltiptext">
                The expected annualized return of the optimized portfolio based on historical data.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Optimized Volatility</div>
            <div class="metric-value">{portfolio_volatility(optimized_weights, returns):.2%}</div>
            <div class="tooltiptext">
                The expected annualized volatility (risk) of the optimized portfolio.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Optimized Sharpe Ratio</div>
            <div class="metric-value">{sharpe_ratio(optimized_weights, returns, risk_free_rate):.2f}</div>
            <div class="tooltiptext">
                The Sharpe Ratio of the optimized portfolio, representing risk-adjusted return.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Display optimized weights
    st.subheader("Optimized Weights")
    st.markdown('''
    <div class="metric-with-tooltip">
        <div class="metric-label">Optimized Weights Table</div>
        <div class="tooltiptext">
            This table shows the asset allocation before and after optimization.
            Optimized weights aim to achieve the selected optimization goal.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    optimized_weights_df = pd.DataFrame({
        'Asset': returns.columns,
        'Original Weight': weights,
        'Optimized Weight': optimized_weights
    })
    st.dataframe(optimized_weights_df.style.format({'Original Weight': '{:.2%}', 'Optimized Weight': '{:.2%}'}))

    # Plot efficient frontier
    st.subheader("Efficient Frontier")

    def efficient_frontier(returns, num_portfolios=1000):
        results = np.zeros((num_portfolios, 3))
        weight_array = []
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            p_ret = portfolio_return(weights, returns)
            p_vol = portfolio_volatility(weights, returns)
            s_ratio = sharpe_ratio(weights, returns, risk_free_rate)
            results[i] = [p_ret, p_vol, s_ratio]
            weight_array.append(weights)
        return results, weight_array

    frontier_data, weight_array = efficient_frontier(returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_data[:, 1],
        y=frontier_data[:, 0],
        mode='markers',
        name='Portfolios',
        marker=dict(
            color=frontier_data[:, 2],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Sharpe Ratio')
        ),
        hovertemplate=(
            'Volatility: %{x:.2%}<br>'
            'Return: %{y:.2%}<br>'
            'Sharpe Ratio: %{marker.color:.2f}<br><br>'
            'This point represents a possible portfolio with the given risk and return characteristics.'
            '<extra></extra>'
        )
    ))
    # Current portfolio marker
    current_vol = portfolio_volatility(weights, returns)
    current_ret = portfolio_return(weights, returns)
    current_sharpe = sharpe_ratio(weights, returns, risk_free_rate)
    fig.add_trace(go.Scatter(
        x=[current_vol],
        y=[current_ret],
        mode='markers',
        name='Current Portfolio',
        marker=dict(size=15, color='red'),
        hovertemplate=(
            'Current Portfolio<br>'
            'Volatility: %{x:.2%}<br>'
            'Return: %{y:.2%}<br>'
            'Sharpe Ratio: ' + f'{current_sharpe:.2f}' + '<br><br>'
            'This point represents your current portfolio based on your asset weights.'
            '<extra></extra>'
        )
    ))
    # Optimized portfolio marker
    opt_vol = portfolio_volatility(optimized_weights, returns)
    opt_ret = portfolio_return(optimized_weights, returns)
    opt_sharpe = sharpe_ratio(optimized_weights, returns, risk_free_rate)
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_ret],
        mode='markers',
        name='Optimized Portfolio',
        marker=dict(size=15, color='green'),
        hovertemplate=(
            'Optimized Portfolio<br>'
            'Volatility: %{x:.2%}<br>'
            'Return: %{y:.2%}<br>'
            'Sharpe Ratio: ' + f'{opt_sharpe:.2f}' + '<br><br>'
            'This point represents the optimized portfolio achieving the selected goal.'
            '<extra></extra>'
        )
    ))
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return',
        hovermode='closest',
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the points to see portfolio metrics. The efficient frontier represents the set of optimal portfolios.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)
