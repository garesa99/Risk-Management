import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.optimize as sco
from openai import OpenAI
import os
client = OpenAI(api_key='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')




# Set page configuration
st.set_page_config(page_title="Advanced Risk Management Dashboard", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .Widget>label {
        color: #fafafa;
        font-weight: bold;
    }
    .stplot {
        background-color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for user inputs and navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Portfolio Overview", "Risk Metrics", "Advanced Analysis", "Stress Testing", "Monte Carlo Simulation", "Factor Analysis", "Portfolio Optimization", "AI Insights", "Raw Data"])

# Function to fetch data (supports multiple asset types)
@st.cache_data
def get_asset_data(assets, start, end):
    data = pd.DataFrame()
    for asset, asset_type in assets:
        if asset_type in ['Stock', 'Crypto']:
            try:
                asset_data = yf.download(asset, start=start, end=end)['Adj Close']
                data[asset] = asset_data
            except Exception as e:
                st.error(f"Error fetching data for {asset}: {e}")
        elif asset_type == 'Custom':
            st.warning(f"Custom data input for {asset} is not implemented yet.")
    return data

# Sidebar inputs
st.sidebar.header("Portfolio Settings")
assets_input = st.sidebar.text_area(
    "Enter assets and quantities (one per line, format: Symbol,Quantity,Type):",
    "AAPL,100,Stock\nMSFT,150,Stock\nGOOGL,75,Stock\nAMZN,50,Stock\nBTC-USD,2,Crypto"
)

# Parse the input
try:
    assets_data = [line.split(',') for line in assets_input.strip().split('\n')]
    assets = [(data[0].strip(), data[2].strip()) for data in assets_data]
    quantities = [float(data[1].strip()) for data in assets_data]
    asset_quantities = pd.Series(quantities, index=[a[0] for a in assets])
except:
    st.error("Invalid input format. Please enter assets as shown in the example.")
    st.stop()

end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)
date_range = st.sidebar.date_input("Select date range", [start_date, end_date])

# Fetch data and calculate returns
df = get_asset_data(assets, date_range[0], date_range[1])
if df.empty:
    st.error("No data available. Please check your inputs and try again.")
    st.stop()

# Calculate portfolio value and weights
portfolio_value = (df.iloc[-1] * asset_quantities).sum()
weights = (df.iloc[-1] * asset_quantities) / portfolio_value

returns = df.pct_change().dropna()
portfolio_returns = (returns * weights).sum(axis=1)
cumulative_returns = (1 + portfolio_returns).cumprod()



# Helper functions
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def calculate_var(returns, confidence_level):
    return np.percentile(returns, 100 - confidence_level)

def calculate_cvar(returns, confidence_level):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result
def get_insights(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def calculate_var(returns, confidence_level):
    return np.percentile(returns, 100 - confidence_level)



# Portfolio Overview Section

if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    
    # Display current portfolio composition
    st.subheader("Current Portfolio Composition")
    composition_df = pd.DataFrame({
        'Asset': [a[0] for a in assets],
        'Type': [a[1] for a in assets],
        'Quantity': asset_quantities,
        'Current Price': df.iloc[-1],
        'Current Value': df.iloc[-1] * asset_quantities,
        'Weight': weights  # Ensure 'Weight' remains numeric here
    })
    
    # Apply background gradient to numeric 'Weight' column
    styled_df = composition_df.style.background_gradient(subset=['Weight'], cmap='viridis')
    
    # Format columns after applying the gradient
    styled_df = styled_df.format({
        'Current Price': '${:.2f}',
        'Current Value': '${:.2f}',
        'Weight': '{:.2%}'
    })
    
    st.table(styled_df)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Portfolio Value", f"${portfolio_value:,.2f}")
        st.metric("Total Return", f"{(cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
    with col2:
        st.metric("Daily Return", f"{portfolio_returns.iloc[-1] * 100:.2f}%")
        st.metric("Annualized Volatility", f"{portfolio_returns.std() * np.sqrt(252) * 100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{(portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252):.2f}")
        st.metric("Sortino Ratio", f"""{(portfolio_returns.mean() - 0.02/252) / 
                    (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)):.2f}""")
    with col4:
        st.metric("Calmar Ratio", f"""{(cumulative_returns.iloc[-1]**(252/len(cumulative_returns))-1) / 
                    abs(portfolio_returns.min()):.2f}""")
        st.metric("Max Drawdown", f"""{(1 - cumulative_returns.div(cumulative_returns.cummax())).max() * 100:.2f}%""")
    
    # Cumulative Returns Chart
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns, 
        mode='lines', 
        name='Portfolio'
    ))
    fig_cumulative.update_layout(
        title='Cumulative Portfolio Returns', 
        xaxis_title='Date', 
        yaxis_title='Cumulative Returns'
    )
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    # Rolling Statistics Chart
    rolling_mean = portfolio_returns.rolling(window=20).mean()
    rolling_std = portfolio_returns.rolling(window=20).std()
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(
        x=portfolio_returns.index, 
        y=portfolio_returns, 
        mode='lines', 
        name='Daily Returns'
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_mean.index, 
        y=rolling_mean, 
        mode='lines', 
        name='20-day Rolling Mean'
    ))
    fig_rolling.add_trace(go.Scatter(
        x=rolling_std.index, 
        y=rolling_std, 
        mode='lines', 
        name='20-day Rolling Std'
    ))
    fig_rolling.update_layout(
        title='Portfolio Returns with Rolling Statistics', 
        xaxis_title='Date', 
        yaxis_title='Returns'
    )
    st.plotly_chart(fig_rolling, use_container_width=True)


elif page == "Risk Metrics":
    st.title("Risk Metrics")
    
    confidence_level = st.slider("Confidence Level for VaR and CVaR", 90, 99, 95)
    var = calculate_var(portfolio_returns, confidence_level)
    cvar = calculate_cvar(portfolio_returns, confidence_level)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Value at Risk ({confidence_level}%)", f"{var*100:.2f}%")
        st.metric("Maximum Drawdown", f"{(1 - cumulative_returns.div(cumulative_returns.cummax())).max()*100:.2f}%")
    with col2:
        st.metric(f"Conditional VaR ({confidence_level}%)", f"{cvar*100:.2f}%")
        st.metric("Beta", f"{stats.linregress(returns.mean(axis=1), portfolio_returns).slope:.2f}")
    with col3:
        try:
            treynor_ratio = (portfolio_returns.mean() - 0.02/252) / stats.linregress(returns.mean(axis=1), portfolio_returns).slope
            st.metric("Treynor Ratio", f"{treynor_ratio:.2f}")
        except:
            st.metric("Treynor Ratio", "N/A (Beta = 0)")
        
        try:
            information_ratio = (portfolio_returns.mean() - returns.mean(axis=1).mean()) / (portfolio_returns - returns.mean(axis=1)).std()
            st.metric("Information Ratio", f"{information_ratio:.2f}")
        except:
            st.metric("Information Ratio", "N/A (No tracking error)")

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name='Daily Returns'))
    fig.add_vline(x=var, line_dash="dash", line_color="red", annotation_text=f"VaR ({confidence_level}%)")
    fig.add_vline(x=cvar, line_dash="dash", line_color="orange", annotation_text=f"CVaR ({confidence_level}%)")
    fig.update_layout(title='Distribution of Daily Returns', xaxis_title='Daily Returns', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Advanced Analysis":
    st.title("Advanced Analysis")
    
    # Augmented Dickey-Fuller test
    def adf_test(series):
        result = adfuller(series.values)
        return pd.Series({'Test Statistic': result[0], 'P-value': result[1], 'Critical Values': result[4]})

    adf_results = adf_test(portfolio_returns)
    st.write("Augmented Dickey-Fuller Test Results:")
    st.write(adf_results)
    st.write("Interpretation: If p-value < 0.05, reject the null hypothesis and conclude the series is stationary.")

    # Autocorrelation plot
    autocorr = [portfolio_returns.autocorr(lag) for lag in range(20)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(20)), y=autocorr))
    fig.update_layout(title='Autocorrelation Plot', xaxis_title='Lag', yaxis_title='Autocorrelation')
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap of correlations
    corr_matrix = returns.corr()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.index, y=corr_matrix.columns, colorscale='Viridis'))
    fig.update_layout(title='Correlation Heatmap', xaxis_title='Assets', yaxis_title='Assets')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Stress Testing":
    st.title("Stress Testing")

    shock_magnitude = st.slider("Shock Magnitude (%)", -50, 50, 0)
    shock_duration = st.slider("Shock Duration (days)", 1, 30, 1)

    def apply_stress(returns, magnitude, duration):
        stressed_returns = returns.copy()
        stress_period = stressed_returns.index[-duration:]
        stressed_returns.loc[stress_period] += magnitude / 100 / duration
        return stressed_returns

    stressed_returns = apply_stress(portfolio_returns, shock_magnitude, shock_duration)
    stressed_cumulative_returns = (1 + stressed_returns).cumprod()

    stressed_total_return = (stressed_cumulative_returns.iloc[-1] - 1) * 100
    stressed_var = calculate_var(stressed_returns, 95)
    stressed_cvar = calculate_cvar(stressed_returns, 95)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stressed Total Return", f"{stressed_total_return:.2f}%", 
                  f"{stressed_total_return - (cumulative_returns.iloc[-1] - 1) * 100:.2f}%")
        st.metric("Stressed VaR (95%)", f"{stressed_var*100:.2f}%", 
                  f"{(stressed_var - calculate_var(portfolio_returns, 95))*100:.2f}%")
    with col2:
        st.metric("Stressed Volatility", f"{stressed_returns.std() * np.sqrt(252) * 100:.2f}%", 
                  f"{(stressed_returns.std() - portfolio_returns.std()) * np.sqrt(252) * 100:.2f}%")
        st.metric("Stressed CVaR (95%)", f"{stressed_cvar*100:.2f}%", 
                  f"{(stressed_cvar - calculate_cvar(portfolio_returns, 95))*100:.2f}%")
    with col3:
        st.metric("Stressed Sharpe Ratio", f"{(stressed_returns.mean() / stressed_returns.std()) * np.sqrt(252):.2f}", 
                  f"{((stressed_returns.mean() / stressed_returns.std()) - (portfolio_returns.mean() / portfolio_returns.std())) * np.sqrt(252):.2f}")
        st.metric("Stressed Max Drawdown", f"{(1 - stressed_cumulative_returns.div(stressed_cumulative_returns.cummax())).max()*100:.2f}%", 
                  f"{((1 - stressed_cumulative_returns.div(stressed_cumulative_returns.cummax())).max() - (1 - cumulative_returns.div(cumulative_returns.cummax())).max())*100:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=stressed_cumulative_returns.index, y=stressed_cumulative_returns, mode='lines', name='Stressed'))
    fig.update_layout(title='Original vs Stressed Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Returns')
    st.plotly_chart(fig, use_container_width=True)

# Monte Carlo Simulation Section
elif page == "Monte Carlo Simulation":
    st.title("Monte Carlo Simulation")

    # User inputs for simulations
    num_simulations = st.slider("Number of Simulations", 1000, 10000, 5000, key="mc_num_simulations")
    time_horizon = st.slider("Time Horizon (trading days)", 30, 252, 252, key="mc_time_horizon")

    # Monte Carlo simulation function
    def monte_carlo_sim(returns, weights, num_simulations, time_horizon):
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        weights = np.array(weights)

        # Generate random returns based on the mean and covariance
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (num_simulations, time_horizon))
        sim_portfolio_returns = np.dot(sim_returns, weights)
        sim_portfolio_cumulative = np.cumprod(1 + sim_portfolio_returns, axis=1)

        return sim_portfolio_cumulative

    # Run the simulation
    sim_results = monte_carlo_sim(returns, weights, num_simulations, time_horizon)

    # Plot the simulation results
    fig = go.Figure()
    for i in range(min(100, num_simulations)):  # Plot first 100 simulations
        fig.add_trace(go.Scatter(
            y=sim_results[i],
            mode='lines',
            line=dict(width=0.5),
            showlegend=False
        ))
    fig.update_layout(
        title=f'Monte Carlo Simulation ({num_simulations} runs, {time_horizon} days)',
        yaxis_title='Cumulative Returns',
        xaxis_title='Trading Days'
    )
    st.plotly_chart(fig, use_container_width=True, key="monte_carlo_chart")

    # Calculate and display statistics
    final_returns = sim_results[:, -1] - 1
    st.write(f"5th Percentile Return: {np.percentile(final_returns, 5):.2%}")
    st.write(f"Median Return: {np.median(final_returns):.2%}")
    st.write(f"95th Percentile Return: {np.percentile(final_returns, 95):.2%}")


elif page == "Factor Analysis":
    st.title("Factor Analysis")

    # Perform PCA
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    pca = PCA()
    pca_results = pca.fit_transform(scaled_returns)

    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_variance_ratio) + 1)), y=explained_variance_ratio, name='Individual'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)), y=cumulative_variance_ratio, mode='lines+markers', name='Cumulative'))
    fig.update_layout(title='PCA Explained Variance Ratio', xaxis_title='Principal Component', yaxis_title='Explained Variance Ratio')
    st.plotly_chart(fig, use_container_width=True)

    # Factor loadings
    num_components = st.slider("Number of Principal Components to Display", 1, len(returns.columns), 3)
    loadings = pd.DataFrame(
        pca.components_.T[:, :num_components],
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=returns.columns
    )
    st.write("Factor Loadings:")
    st.dataframe(loadings.style.background_gradient(cmap='viridis'))

    # Biplot of first two principal components
    if len(returns.columns) > 2:
        fig = go.Figure()
        for i, asset in enumerate(returns.columns):
            fig.add_trace(go.Scatter(
                x=[0, pca.components_[0, i]],
                y=[0, pca.components_[1, i]],
                mode='lines+markers+text',
                name=asset,
                text=[None, asset],
                textposition="top center"
            ))
        fig.update_layout(
            title='PCA Biplot',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Portfolio Optimization":
    st.title("Portfolio Optimization")

    risk_free_rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 5.0, 1.0) / 100

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Current portfolio performance
    current_std, current_return = portfolio_performance(weights, mean_returns, cov_matrix)
    current_sharpe = (current_return - risk_free_rate) / current_std

    # Optimize portfolio
    optimized = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    
    sdp, rp = portfolio_performance(optimized['x'], mean_returns, cov_matrix)
    optimized_sharpe = (rp - risk_free_rate) / sdp

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Portfolio")
        st.metric("Return", f"{current_return*100:.2f}%")
        st.metric("Volatility", f"{current_std*100:.2f}%")
        st.metric("Sharpe Ratio", f"{current_sharpe:.2f}")

    with col2:
        st.subheader("Optimized Portfolio")
        st.metric("Return", f"{rp*100:.2f}%", f"{(rp - current_return)*100:.2f}%")
        st.metric("Volatility", f"{sdp*100:.2f}%", f"{(sdp - current_std)*100:.2f}%")
        st.metric("Sharpe Ratio", f"{optimized_sharpe:.2f}", f"{optimized_sharpe - current_sharpe:.2f}")

    # Display optimized weights
    st.subheader("Optimized Weights")
    opt_weights = pd.Series(optimized['x'], index=[a[0] for a in assets])
    fig = go.Figure(go.Bar(x=opt_weights.index, y=opt_weights.values))
    fig.update_layout(title='Optimized Portfolio Weights', xaxis_title='Asset', yaxis_title='Weight')
    st.plotly_chart(fig, use_container_width=True)

    # Efficient Frontier
    st.subheader("Efficient Frontier")

    def efficient_frontier(mean_returns, cov_matrix, returns_range):
        results = []
        for ret in returns_range:
            cons = ({'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1] - ret},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            result = sco.minimize(lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0], 
                                  len(mean_returns)*[1./len(mean_returns),], method='SLSQP', bounds=tuple((0,1) for _ in range(len(mean_returns))),
                                  constraints=cons)
            results.append(result)
        return results

    returns_range = np.linspace(min(mean_returns), max(mean_returns), 50)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, returns_range)

    volatilities = [p['fun'] for p in efficient_portfolios]
    returns = [portfolio_performance(p['x'], mean_returns, cov_matrix)[1] for p in efficient_portfolios]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=volatilities, y=returns, mode='lines', name='Efficient Frontier'))
    fig.add_trace(go.Scatter(x=[current_std], y=[current_return], mode='markers', name='Current Portfolio', marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(x=[sdp], y=[rp], mode='markers', name='Optimized Portfolio', marker=dict(size=10, color='green')))
    fig.update_layout(title='Efficient Frontier', xaxis_title='Annualized Volatility', yaxis_title='Annualized Return')
    st.plotly_chart(fig, use_container_width=True)



######################AI SECTIOM

elif page == "AI Insights":
    st.title("AI Insights and Recommendations")

    # Calculate VaR here
    confidence_level = 95  # You can adjust this as needed
    var = calculate_var(portfolio_returns, confidence_level)

    # Example: Portfolio Summary Insights
    portfolio_summary_prompt = f"Summarize the performance of a portfolio with a total value of ${portfolio_value:,.2f}, a return of {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%, and a Sharpe ratio of {(portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252):.2f}. Mention the key metrics and risks."
    portfolio_summary = get_insights(portfolio_summary_prompt)
    st.subheader("Portfolio Summary")
    st.write(portfolio_summary)

    # Example: Investment Recommendations
    investment_recommendation_prompt = "Suggest assets that can help diversify a tech-heavy portfolio consisting of AAPL, MSFT, and GOOGL."
    investment_recommendations = get_insights(investment_recommendation_prompt)
    st.subheader("Investment Recommendations")
    st.write(investment_recommendations)

    # Example: Risk Management Suggestions
    risk_management_prompt = f"Given a portfolio with an annualized volatility of {portfolio_returns.std() * np.sqrt(252) * 100:.2f}% and a VaR ({confidence_level}%) of {var*100:.2f}%, suggest strategies to reduce risk."
    risk_management_suggestions = get_insights(risk_management_prompt)
    st.subheader("Risk Management Suggestions")
    st.write(risk_management_suggestions)

    # Example: Market Insights
    market_insights_prompt = "Provide insights on recent trends in the tech and crypto markets that could impact AAPL, MSFT, GOOGL, and BTC-USD."
    market_insights = get_insights(market_insights_prompt)
    st.subheader("Market Insights")
    st.write(market_insights)




elif page == "Raw Data":
    st.title("Raw Data")
    st.dataframe(df)
    
    st.subheader("Returns Data")
    st.dataframe(returns)

    st.subheader("Portfolio Composition")
    st.dataframe(pd.DataFrame({
        'Asset': [a[0] for a in assets],
        'Type': [a[1] for a in assets],
        'Quantity': asset_quantities,
        'Current Price': df.iloc[-1],
        'Current Value': df.iloc[-1] * asset_quantities,
        'Weight': weights
    }))

# Update the sidebar info
st.sidebar.info("Created by [Gabriel Reyes]")
st.sidebar.text("v1.3.0 - Multi-Asset Support")