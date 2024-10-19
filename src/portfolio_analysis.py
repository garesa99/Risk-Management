import numpy as np
import pandas as pd

def calculate_portfolio_returns(returns, weights):
    return (returns * weights).sum(axis=1)

def calculate_metrics(portfolio_returns, portfolio_value):
    cumulative_returns = (1 + portfolio_returns).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) * 100
    daily_return = portfolio_returns.iloc[-1] * 100
    annualized_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
    sortino_ratio = (portfolio_returns.mean() - 0.02/252) / (portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252))
    calmar_ratio = (cumulative_returns.iloc[-1]**(252/len(cumulative_returns))-1) / abs(portfolio_returns.min())
    max_drawdown = (1 - cumulative_returns / cumulative_returns.cummax()).max() * 100

    return {
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'daily_return': daily_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown
    }

def calculate_var(returns, confidence_level):
    return np.percentile(returns, 100 - confidence_level)

def calculate_cvar(returns, confidence_level):
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns