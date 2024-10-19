import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.portfolio_analysis import calculate_portfolio_returns

st.set_page_config(layout="wide", page_title="Monte Carlo Simulation")

st.title("🎲 Monte Carlo Simulation")

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

    st.subheader("Monte Carlo Simulation Parameters")

    num_simulations = st.slider("Number of Simulations", 100, 10000, 1000)
    time_horizon = st.slider("Time Horizon (days)", 30, 252, 252)

    # Run Monte Carlo simulation
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()

    simulation_df = pd.DataFrame()

    for i in range(num_simulations):
        daily_returns = np.random.normal(mean_return, std_return, time_horizon)
        cumulative_return = np.cumprod(1 + daily_returns)
        simulation_df[f'Sim {i+1}'] = cumulative_return

    # Plot results
    fig = go.Figure()
    for i in range(min(100, num_simulations)):  # Plot first 100 simulations
        fig.add_trace(go.Scatter(
            y=simulation_df[f'Sim {i+1}'],
            mode='lines',
            name=f'Sim {i+1}',
            opacity=0.3,
            hovertemplate=f'Simulation {i+1}<br>Day: %{{x}}<br>Cumulative Return: %{{y:.2f}}x<extra></extra>'
        ))

    fig.update_layout(
        title=f'Monte Carlo Simulation: {num_simulations} runs over {time_horizon} days',
        xaxis_title='Days',
        yaxis_title='Cumulative Returns (Growth of $1)',
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the lines to see the cumulative return for each simulation at different time points.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate and display statistics
    final_cumulative_returns = simulation_df.iloc[-1]

    st.subheader("Simulation Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        percentile_5 = np.percentile(final_cumulative_returns, 5)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">5th Percentile</div>
            <div class="metric-value">{percentile_5:.2f}x</div>
            <div class="tooltiptext">
                The 5th percentile represents a pessimistic outcome where only 5% of simulations performed worse. 
                This can be interpreted as a lower bound in a worst-case scenario.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        median = np.median(final_cumulative_returns)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">Median</div>
            <div class="metric-value">{median:.2f}x</div>
            <div class="tooltiptext">
                The median represents the middle value of the simulations, indicating a typical expected outcome.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        percentile_95 = np.percentile(final_cumulative_returns, 95)
        st.markdown(f'''
        <div class="metric-with-tooltip">
            <div class="metric-label">95th Percentile</div>
            <div class="metric-value">{percentile_95:.2f}x</div>
            <div class="tooltiptext">
                The 95th percentile represents an optimistic outcome where only 5% of simulations performed better. 
                This can be seen as an upper bound in a best-case scenario.
            </div>
        </div>
        ''', unsafe_allow_html=True)
