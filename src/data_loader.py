import yfinance as yf
import pandas as pd
import streamlit as st

def load_data(assets_input, start_date, end_date):
    
    try:
        assets_data = [line.split(',') for line in assets_input.strip().split('\n')]
        assets = [(data[0].strip(), data[2].strip()) for data in assets_data]
        quantities = [float(data[1].strip()) for data in assets_data]
        asset_quantities = pd.Series(quantities, index=[a[0] for a in assets])
    except Exception as e:
        st.error(f"Error parsing input: {str(e)}")
        raise


    df = pd.DataFrame()
    for asset, asset_type in assets:
        if asset_type in ['Stock', 'Crypto']:
            try:
                ticker = yf.Ticker(asset)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty and 'Close' in hist.columns:
                    df[asset] = hist['Close']  
                    st.success(f"Successfully fetched data for {asset}")
                else:
                    st.warning(f"No data available for {asset} in the specified date range.")
            except Exception as e:
                st.error(f"Error fetching data for {asset}: {str(e)}")
        else:
            st.warning(f"Custom data input for {asset} is not implemented yet.")

    if df.empty:
        st.error("No data available. Please check your inputs and try again.")
        raise ValueError("No data available")

    
    try:
        portfolio_value = (df.iloc[-1] * asset_quantities).sum()
        weights = (df.iloc[-1] * asset_quantities) / portfolio_value
    except Exception as e:
        st.error(f"Error calculating portfolio value and weights: {str(e)}")
        raise

    returns = df.pct_change().dropna()
    
    return df, returns, weights, portfolio_value
