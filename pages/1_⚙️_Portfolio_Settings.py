import streamlit as st
import pandas as pd
import datetime
import altair as alt
from src.data_loader import load_data

st.set_page_config(layout="wide", page_title="Portfolio Settings")

st.title("⚙️ Portfolio Settings")

# Add custom CSS for tooltips
st.markdown("""
<style>
.metric-with-tooltip {
    position: relative;
    display: inline-block;
    margin: 10px;
    text-align: center;
}

.metric-with-tooltip .tooltiptext {
    visibility: hidden;
    width: 220px;
    background-color: #555;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position above */
    left: 50%;
    margin-left: -110px;
    opacity: 0;
    transition: opacity 0.3s;
}

.metric-with-tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

.metric-label {
    font-size: 16px;
    color: #666;
}

.metric-value {
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def validate_input(assets_input):
    lines = assets_input.strip().split('\n')
    for line in lines:
        parts = line.split(',')
        if len(parts) != 3 or not parts[1].replace('.', '').isdigit():
            return False
    return True

# Portfolio input form
with st.form("portfolio_input_form"):
    st.subheader("Enter Your Portfolio Details")
    assets_input = st.text_area(
        "Enter assets (Symbol,Quantity,Type):",
        value=st.session_state.get('assets_input', "AAPL,100,Stock\nMSFT,150,Stock\nBTC-USD,2,Crypto"),
        help="Enter one asset per line in the format: Symbol,Quantity,Type"
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get('start_date', datetime.date.today() - datetime.timedelta(days=365))
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get('end_date', datetime.date.today())
        )

    submit_button = st.form_submit_button("Load Portfolio Data")

    if submit_button:
        if not validate_input(assets_input):
            st.error("Invalid input format. Please enter assets in the format: Symbol,Quantity,Type")
        elif start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            try:
                with st.spinner("Loading data... This may take a moment."):
                    df, returns, weights, portfolio_value = load_data(assets_input, start_date, end_date)
                
                if df.empty or returns.empty or len(weights) == 0:
                    st.error("No data available for the selected assets and date range. Please try different inputs.")
                else:
                    st.session_state.update(dict(
                        df=df, returns=returns, weights=weights,
                        portfolio_value=portfolio_value,
                        assets_input=assets_input, start_date=start_date, end_date=end_date,
                        data_loaded=True
                    ))
                    st.success("Portfolio data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# Display loaded data summary
if st.session_state.get('data_loaded', False):
    st.subheader("Portfolio Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'''
        <div class="metric-with-tooltip">
          <div class="metric-label">Number of Assets</div>
          <div class="metric-value">{len(st.session_state.weights)}</div>
          <div class="tooltiptext">The total number of different assets in your portfolio.</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="metric-with-tooltip">
          <div class="metric-label">Date Range</div>
          <div class="metric-value">{st.session_state.start_date} to {st.session_state.end_date}</div>
          <div class="tooltiptext">The time period over which your portfolio data is analyzed.</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="metric-with-tooltip">
          <div class="metric-label">Total Portfolio Value</div>
          <div class="metric-value">${st.session_state.portfolio_value:,.2f}</div>
          <div class="tooltiptext">The total monetary value of your portfolio based on current asset prices.</div>
        </div>
        ''', unsafe_allow_html=True)

    st.subheader("Asset Allocation")
    allocation_df = pd.DataFrame({
        'Asset': st.session_state.weights.index,
        'Weight': st.session_state.weights.values,
        'Value': st.session_state.weights * st.session_state.portfolio_value
    })
    allocation_df = allocation_df.sort_values('Value', ascending=False)
    allocation_df['Weight (%)'] = allocation_df['Weight'] * 100

    # Add an explanation column for the tooltip
    allocation_df['Explanation'] = 'This bar represents the monetary value allocated to each asset.'

    # Create an interactive bar chart with tooltips
    chart = alt.Chart(allocation_df).mark_bar().encode(
        x=alt.X('Asset:N', sort=None, title='Asset'),
        y=alt.Y('Value:Q', title='Value in USD'),
        tooltip=[
            alt.Tooltip('Asset:N', title='Asset'),
            alt.Tooltip('Value:Q', title='Value', format='$,.2f'),
            alt.Tooltip('Weight (%):Q', title='Weight (%)', format='.2f'),
            alt.Tooltip('Explanation:N', title='Explanation')
        ]
    ).properties(
        width='container',
        height=400,
        title='Asset Allocation by Value'
    )

    st.altair_chart(chart, use_container_width=True)

    st.caption("Hover over the bars to see detailed information about each asset.")

# Add a button to clear the portfolio data
if st.session_state.get('data_loaded', False):
    if st.button("Clear Portfolio Data"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Portfolio data cleared. You can now load new data.")
        st.experimental_rerun()

# Information message if no data is loaded
if not st.session_state.get('data_loaded', False):
    st.info("No portfolio data is currently loaded. Please enter your portfolio details above and click 'Load Portfolio Data'.")
