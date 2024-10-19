import numpy as np
import pandas as pd
from src.portfolio_analysis import calculate_metrics, calculate_var, calculate_cvar

def apply_stress(returns, magnitude, duration):
    stressed_returns = returns.copy()
    stress_period = stressed_returns.index[-duration:]
    stressed_returns.loc[stress_period] += magnitude / 100 / duration
    return stressed_returns

def stress_test_portfolio(returns, weights, magnitude, duration):
    portfolio_returns = (returns * weights).sum(axis=1)
    stressed_returns = apply_stress(portfolio_returns, magnitude, duration)
    stressed_cumulative_returns = (1 + stressed_returns).cumprod()

    original_metrics = calculate_metrics(portfolio_returns, 1)  # Assuming initial portfolio value of 1
    stressed_metrics = calculate_metrics(stressed_returns, 1)

    var_original = calculate_var(portfolio_returns, 95)
    var_stressed = calculate_var(stressed_returns, 95)
    cvar_original = calculate_cvar(portfolio_returns, 95)
    cvar_stressed = calculate_cvar(stressed_returns, 95)

    results = {
        'Original': original_metrics,
        'Stressed': stressed_metrics,
        'VaR_Original': var_original,
        'VaR_Stressed': var_stressed,
        'CVaR_Original': cvar_original,
        'CVaR_Stressed': cvar_stressed
    }

    return results, stressed_cumulative_returns