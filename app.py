import streamlit as st
import datetime
from src.data_loader import load_data
from pages import (
    show_portfolio_settings, show_overview, show_risk_metrics, 
    show_advanced_analysis, show_stress_testing, show_monte_carlo, 
    show_factor_analysis, show_optimization, show_ai_insights, show_raw_data
)

# Clear the cache
st.cache_data.clear()

def show_welcome():
    st.title("Welcome to the Advanced Risk Management Dashboard")
    st.header("Comprehensive Portfolio Analysis Tools")
    
    st.markdown("""
    This dashboard provides advanced tools for portfolio analysis and risk management. Here's what you can do:
    """)

    # Features overview with emojis
    features = {
        "⚙️ Portfolio Settings": "Configure your initial setup. Input your assets, quantities, and investment period here.",
        "📊 Portfolio Overview": "View your portfolio composition, performance metrics, and dynamic visualizations.",
        "📈 AI Insights": "Get personalized AI recommendations and market trend analysis for your portfolio.",
        "📉 Risk Metrics": "Explore key risk indicators including VaR, CVaR, beta, and other essential metrics.",
        "📊 Advanced Analysis": "Access sophisticated tools for in-depth portfolio performance evaluation.",
        "🎲 Monte Carlo Simulation": "Visualize potential future scenarios and understand probability distributions.",
        "🔥 Stress Testing": "Test how your portfolio might perform under various market conditions.",
        "📈 Portfolio Optimization": "Discover optimal asset allocations based on your risk-return preferences.",
        "🔍 Factor Analysis": "Understand the key drivers affecting your portfolio's performance.",
        "📑 Raw Data": "Download and examine the underlying data for your own analysis."
    }

    # Display features in a clean format
    for feature, description in features.items():
        st.markdown(f"**{feature}**")
        st.markdown(description)
        st.markdown("---")

    # Getting Started Section
    st.header("Getting Started")
    st.markdown("""
    1. Start with **Portfolio Settings** to input your portfolio data
    2. Follow the simple format: Symbol,Quantity,Type (Example: AAPL,100,Stock)
    3. Select your preferred analysis timeframe
    4. Explore different analysis tools using the navigation menu
    """)

    # Display version and creator info
    st.sidebar.markdown("---")
    st.sidebar.info("Created by Gabriel Reyes\nContact: gabriel.reyes@gsom.polimi.it")
    st.sidebar.text("v1.3.0 - Multi-Asset Support")

def main():
    st.set_page_config(page_title="Advanced Risk Management Dashboard", layout="wide")
    
    # Apply custom CSS
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

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Welcome",
        "Portfolio Settings",
        "Portfolio Overview",
        "AI Insights",
        "Risk Metrics",
        "Advanced Analysis",
        "Monte Carlo Simulation",
        "Stress Testing",
        "Portfolio Optimization",
        "Factor Analysis",
        "Raw Data"
    ])

    # Route to the appropriate page
    if page == "Welcome":
        show_welcome()
    elif page == "Portfolio Settings":
        show_portfolio_settings()
    elif page == "Portfolio Overview":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_overview(st.session_state.get('df'), 
                     st.session_state.get('returns'),
                     st.session_state.get('weights'),
                     st.session_state.get('portfolio_value'))
    elif page == "Risk Metrics":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_risk_metrics(st.session_state.get('returns'),
                         st.session_state.get('weights'))
    elif page == "Advanced Analysis":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_advanced_analysis(st.session_state.get('returns'))
    elif page == "Stress Testing":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_stress_testing(st.session_state.get('returns'),
                          st.session_state.get('weights'))
    elif page == "Monte Carlo Simulation":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_monte_carlo(st.session_state.get('returns'),
                        st.session_state.get('weights'))
    elif page == "Factor Analysis":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_factor_analysis(st.session_state.get('returns'))
    elif page == "Portfolio Optimization":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_optimization(st.session_state.get('returns'),
                        st.session_state.get('weights'))
    elif page == "AI Insights":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_ai_insights(st.session_state.get('df'),
                        st.session_state.get('returns'),
                        st.session_state.get('weights'),
                        st.session_state.get('portfolio_value'))
    elif page == "Raw Data":
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load your portfolio data in Portfolio Settings first.")
            return
        show_raw_data(st.session_state.get('df'),
                     st.session_state.get('returns'),
                     st.session_state.get('weights'))

if __name__ == "__main__":
    main()