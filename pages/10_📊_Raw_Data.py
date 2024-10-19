import streamlit as st
import pandas as pd

st.set_page_config(layout="wide", page_title="Raw Data")

st.title("📊 Raw Data")

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    df = st.session_state.df
    returns = st.session_state.returns
    weights = st.session_state.weights
    
    st.subheader("Price Data")
    st.dataframe(df)
    
    st.download_button(
        label="Download Price Data as CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='price_data.csv',
        mime='text/csv',
    )
    
    st.subheader("Returns Data")
    st.dataframe(returns)
    
    st.download_button(
        label="Download Returns Data as CSV",
        data=returns.to_csv().encode('utf-8'),
        file_name='returns_data.csv',
        mime='text/csv',
    )
    
    st.subheader("Current Portfolio Weights")
    weights_df = pd.DataFrame({'Asset': weights.index, 'Weight': weights})
    st.dataframe(weights_df)
    
    st.download_button(
        label="Download Weights Data as CSV",
        data=weights_df.to_csv().encode('utf-8'),
        file_name='weights_data.csv',
        mime='text/csv',
    )
    
    st.subheader("Data Statistics")
    st.write("Price Data Statistics:")
    st.dataframe(df.describe())
    
    st.write("Returns Data Statistics:")
    st.dataframe(returns.describe())