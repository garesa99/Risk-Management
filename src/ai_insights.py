import openai
from config import API_KEY

# Set your OpenAI API key
openai.api_key = API_KEY

def get_insights(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

def generate_portfolio_summary(portfolio_value, total_return, sharpe_ratio):
    prompt = f"Summarize the performance of a portfolio with a total value of ${portfolio_value:,.2f}, a return of {total_return:.2f}%, and a Sharpe ratio of {sharpe_ratio:.2f}. Mention the key metrics and risks."
    return get_insights(prompt)

def generate_investment_recommendations(assets):
    prompt = f"Suggest assets that can help diversify a portfolio consisting of {', '.join(assets)}."
    return get_insights(prompt)

def generate_risk_management_suggestions(volatility, var):
    prompt = f"Given a portfolio with an annualized volatility of {volatility:.2f}% and a VaR (95%) of {var:.2f}%, suggest strategies to reduce risk."
    return get_insights(prompt)

def generate_market_insights(assets):
    prompt = f"Provide insights on recent trends in the markets that could impact {', '.join(assets)}."
    return get_insights(prompt)
