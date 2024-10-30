import streamlit as st

st.set_page_config(page_title="Advanced Risk Management Dashboard", layout="wide")

st.title("Welcome to the Risk Management Dashboard")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50; /* Dark blue */
        font-weight: 700;
        font-family: 'Arial', sans-serif;
    }
    .feature-header {
        font-size: 1.5rem;
        color: #2C3E50; /* Dark blue */
        font-weight: 600;
        font-family: 'Arial', sans-serif;
    }
    .feature-text {
        font-size: 1rem;
        color: #4D4D4D; /* Dark gray */
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Comprehensive Portfolio Analysis Tools</p>', unsafe_allow_html=True)

st.write("""
This dashboard provides advanced tools for portfolio analysis and risk management.
Use the navigation menu above to explore different features:
""")

features = [
    
    ("Portfolio Settings", "First, go to Portfolio Settings and load your portfolio Data."),
    ("Portfolio Overview", "Get a snapshot of your current portfolio composition and performance metrics."),
    ("Risk Metrics", "Analyze various risk measures including VaR, CVaR, and more."),
    ("Advanced Analysis", "Dive deeper into your portfolio's performance with sophisticated analytical tools."),
    ("Stress Testing", "Evaluate your portfolio's performance under different market scenarios."),
    ("Monte Carlo Simulation", "Project potential future outcomes for your portfolio using probabilistic modeling."),
    ("Factor Analysis", "Understand the driving factors influencing your portfolio's performance."),
    ("Portfolio Optimization", "Find optimal asset allocations to maximize returns or minimize risk."),
    ("AI Insights", "Leverage artificial intelligence to gain unique insights about your portfolio."),
    ("Raw Data", "Access and examine the underlying data used in all analyses.")
]

for feature, description in features:
    st.markdown(f'<p class="feature-header">{feature}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="feature-text">{description}</p>', unsafe_allow_html=True)

st.sidebar.info("Created by [Gabriel Reyes - gabriel.reyes@gsom.polimi.it]")
st.sidebar.text("v1.3.0 - Multi-Asset Support")
