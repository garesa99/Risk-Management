import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Factor Analysis")

st.title("🔬 Factor Analysis")

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
    width: 240px;
    background-color: #555;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Position above */
    left: 50%;
    margin-left: -120px;
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

if not st.session_state.get('data_loaded', False):
    st.warning("Please load your portfolio data in the Portfolio Settings page first.")
else:
    returns = st.session_state.returns

    st.subheader("Correlation Analysis")

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    # Plot correlation heatmap with hover tooltips
    heatmap = go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hovertemplate='Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.2f}<br><br>This value represents the correlation between the two assets.<extra></extra>'
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title='Asset Correlation Heatmap',
        xaxis_title='Assets',
        yaxis_title='Assets',
        margin=dict(l=60, r=60, t=60, b=60)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.1,
        xref='paper',
        yref='paper',
        text='Hover over the heatmap to see the correlation between assets.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)

 
    st.markdown(f'''
    <div class="metric-with-tooltip">
        <div class="metric-label">Correlation Matrix</div>
        <div class="tooltiptext">
            The correlation matrix shows the pairwise correlation coefficients between assets in your portfolio.
            Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.dataframe(corr_matrix.style.format("{:.2f}"))

    st.subheader("Rolling Correlation Analysis")


    assets = st.multiselect(
        "Select assets for rolling correlation",
        options=returns.columns,
        default=returns.columns[:2]
    )

    if len(assets) == 2:
        window = st.slider("Rolling window (days)", min_value=5, max_value=252, value=30)

        rolling_corr = returns[assets].rolling(window=window).corr().unstack()[assets[0]][assets[1]]

        fig = go.Figure(
            data=go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode='lines',
                hovertemplate='Date: %{x}<br>Rolling Correlation: %{y:.2f}<br><br>This line represents the rolling correlation between the two selected assets over the specified window.<extra></extra>'
            )
        )
        fig.update_layout(
            title=f'{window}-day Rolling Correlation: {assets[0]} vs {assets[1]}',
            xaxis_title='Date',
            yaxis_title='Correlation',
            hovermode='x unified',
            margin=dict(l=60, r=60, t=60, b=60)
        )
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref='paper',
            yref='paper',
            text='Hover over the line to see the rolling correlation at different dates.',
            showarrow=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select exactly two assets for rolling correlation analysis.")

    st.subheader("Principal Component Analysis (PCA)")

    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(returns.fillna(0))  

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(explained_variance_ratio) + 1)),
        y=explained_variance_ratio,
        name='Individual',
        hovertemplate='Principal Component %{x}<br>Explained Variance Ratio: %{y:.2%}<br><br>This bar shows the variance explained by each principal component.<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance_ratio) + 1)),
        y=cumulative_variance_ratio,
        mode='lines+markers',
        name='Cumulative',
        hovertemplate='Principal Component %{x}<br>Cumulative Explained Variance: %{y:.2%}<br><br>This line represents the cumulative variance explained up to each principal component.<extra></extra>'
    ))
    fig.update_layout(
        title='Explained Variance Ratio',
        xaxis_title='Principal Component',
        yaxis_title='Explained Variance Ratio',
        hovermode='x unified',
        margin=dict(l=60, r=60, t=60, b=60)
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref='paper',
        yref='paper',
        text='Hover over the bars and line to see how much variance each component explains.',
        showarrow=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factor Loadings")
    num_components = st.slider("Number of Components to Display", 1, len(returns.columns), 3)
    loadings = pd.DataFrame(
        pca.components_[:num_components].T,
        columns=[f'PC{i+1}' for i in range(num_components)],
        index=returns.columns
    )
    st.markdown(f'''
    <div class="metric-with-tooltip">
        <div class="metric-label">Factor Loadings Table</div>
        <div class="tooltiptext">
            Factor loadings represent the correlation coefficients between the original variables and the principal components.
            They indicate how much each asset contributes to the principal components.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    st.dataframe(loadings.style.format("{:.4f}"))
