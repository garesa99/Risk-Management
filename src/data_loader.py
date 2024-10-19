import yfinance as yf
import pandas as pd
import streamlit as st

def load_data(assets_input, start_date, end_date):
    st.write("Debug: Entering load_data function")
    st.write(f"Debug: Assets input: {assets_input}")
    st.write(f"Debug: Start date: {start_date}, End date: {end_date}")

    # Parse the input
    try:
        assets_data = [line.split(',') for line in assets_input.strip().split('\n')]
        assets = [(data[0].strip(), data[2].strip()) for data in assets_data]
        quantities = [float(data[1].strip()) for data in assets_data]
        asset_quantities = pd.Series(quantities, index=[a[0] for a in assets])
        st.write("Debug: Parsed assets:", assets)
        st.write("Debug: Parsed quantities:", quantities)
    except Exception as e:
        st.error(f"Error parsing input: {str(e)}")
        raise

    # Fetch data
    df = pd.DataFrame()
    for asset, asset_type in assets:
        if asset_type in ['Stock', 'Crypto']:
            try:
                st.write(f"Debug: Fetching data for {asset}")
                asset_data = yf.download(asset, start=start_date, end=end_date)['Adj Close']
                if asset_data.empty:
                    st.warning(f"No data available for {asset} in the specified date range.")
                else:
                    df[asset] = asset_data
                    st.write(f"Debug: Data fetched for {asset}, shape: {asset_data.shape}")
            except Exception as e:
                st.error(f"Error fetching data for {asset}: {str(e)}")
        elif asset_type == 'Custom':
            st.warning(f"Custom data input for {asset} is not implemented yet.")

    if df.empty:
        st.error("No data available. Please check your inputs and try again.")
        raise ValueError("No data available")

    # Calculate portfolio value and weights
    try:
        st.write("Debug: Calculating portfolio value and weights")
        portfolio_value = (df.iloc[-1] * asset_quantities).sum()
        weights = (df.iloc[-1] * asset_quantities) / portfolio_value
        st.write(f"Debug: Portfolio value: {portfolio_value}")
        st.write(f"Debug: Weights: {weights}")
    except Exception as e:
        st.error(f"Error calculating portfolio value and weights: {str(e)}")
        raise

    returns = df.pct_change().dropna()
    st.write(f"Debug: Returns shape: {returns.shape}")

    return df, returns, weights, portfolio_value