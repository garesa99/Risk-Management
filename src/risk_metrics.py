import numpy as np

def calculate_beta(portfolio_returns, market_returns):
    """Calculate Beta"""
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance

def calculate_treynor_ratio(portfolio_returns, market_returns, risk_free_rate):
    """Calculate Treynor Ratio"""
    beta = calculate_beta(portfolio_returns, market_returns)
    return (portfolio_returns.mean() - risk_free_rate) / beta

def calculate_information_ratio(portfolio_returns, benchmark_returns):
    """Calculate Information Ratio"""
    return (portfolio_returns.mean() - benchmark_returns.mean()) / (portfolio_returns - benchmark_returns).std()

def calculate_downside_deviation(returns, target=0):
    """Calculate Downside Deviation"""
    downside_returns = returns[returns < target]
    return np.sqrt(np.mean(downside_returns**2))

def calculate_sortino_ratio(returns, risk_free_rate, target=0):
    """Calculate Sortino Ratio"""
    excess_return = returns.mean() - risk_free_rate
    downside_dev = calculate_downside_deviation(returns, target)
    return excess_return / downside_dev

def calculate_max_drawdown(returns):
    """Calculate Maximum Drawdown"""
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

def calculate_var(returns, confidence_level):
    """Calculate Value at Risk"""
    return np.percentile(returns, 100 - confidence_level)

def calculate_cvar(returns, confidence_level):
    """Calculate Conditional Value at Risk"""
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()