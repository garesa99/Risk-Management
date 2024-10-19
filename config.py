import os
import streamlit as st

# For local development, use environment variables
API_KEY = os.getenv("API_KEY")

# For Streamlit Cloud, use Streamlit Secrets Management
if st.secrets.get("API_KEY"):
    API_KEY = st.secrets["API_KEY"]



DEFAULT_ASSETS = "AAPL,100,Stock\nMSFT,150,Stock\nGOOGL,75,Stock\nAMZN,50,Stock\nBTC-USD,2,Crypto"

