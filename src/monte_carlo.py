import numpy as np
import pandas as pd

def monte_carlo_sim(returns, weights, num_simulations, time_horizon):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    weights = np.array(weights)

    # Generate random returns
    sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, (num_simulations, time_horizon))
    sim_portfolio_returns = np.dot(sim_returns, weights)
    sim_portfolio_cumulative = np.cumprod(1 + sim_portfolio_returns, axis=1)

    return sim_portfolio_cumulative

def calculate_var(simulated_returns, confidence_level):
    return np.percentile(simulated_returns, 100 - confidence_level)

def calculate_cvar(simulated_returns, confidence_level):
    var = calculate_var(simulated_returns, confidence_level)
    return simulated_returns[simulated_returns <= var].mean()

def summarize_monte_carlo(simulated_returns):
    final_returns = simulated_returns[:, -1] - 1
    summary = {
        '5th Percentile': np.percentile(final_returns, 5),
        'Median': np.median(final_returns),
        '95th Percentile': np.percentile(final_returns, 95)
    }
    return summary