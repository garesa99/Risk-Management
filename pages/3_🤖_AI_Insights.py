import streamlit as st
import pandas as pd
import numpy as np
import openai
from config import API_KEY

# Set up OpenAI API
openai.api_key = st.secrets["API_KEY"]

st.set_page_config(layout="wide", page_title="AI Insights")

st.title("🤖 AI Insights")

def generate_insight(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error generating insight: {str(e)}"

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    returns = st.session_state.returns
    weights = st.session_state.weights
    df = st.session_state.df
    portfolio_value = st.session_state.portfolio_value  # Use the portfolio value from session state
    
    # Calculate portfolio statistics
    portfolio_returns = (returns * weights).sum(axis=1)
    total_return = (portfolio_returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    
    # Prepare portfolio summary for AI
    portfolio_summary = f"""
    Portfolio Summary:
    - Total Value: ${portfolio_value:,.2f}
    - Number of Assets: {len(weights)}
    - Top 3 Holdings: {', '.join(weights.nlargest(3).index)}
    - Annualized Return: {annualized_return:.2%}
    - Annualized Volatility: {annualized_volatility:.2%}
    - Sharpe Ratio: {sharpe_ratio:.2f}
    """
    
    st.subheader("Portfolio Composition Analysis")
    composition_prompt = f"Based on this portfolio summary, provide insights on the portfolio composition and suggest potential improvements:\n\n{portfolio_summary}"
    composition_insight = generate_insight(composition_prompt)
    st.write(composition_insight)
    
    st.subheader("Performance Insights")
    performance_prompt = f"Analyze the performance of this portfolio and provide insights:\n\n{portfolio_summary}"
    performance_insight = generate_insight(performance_prompt)
    st.write(performance_insight)
    
    st.subheader("Risk Analysis")
    risk_prompt = f"Provide a risk analysis for this portfolio and suggest risk management strategies:\n\n{portfolio_summary}"
    risk_insight = generate_insight(risk_prompt)
    st.write(risk_insight)
    
    st.subheader("Market Trends and Recommendations")
    market_prompt = "Based on current market trends, provide recommendations for portfolio adjustments and potential opportunities."
    market_trends = generate_insight(market_prompt)
    st.write(market_trends)
    
    st.subheader("Customized Actions")
    actions_prompt = f"Based on this portfolio summary, suggest 3-5 specific actions the investor should consider:\n\n{portfolio_summary}"
    actions = generate_insight(actions_prompt)
    st.write(actions)

    st.info("Note: These insights are generated by an AI model and should be used in conjunction with professional financial advice.")