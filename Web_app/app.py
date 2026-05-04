import pandas as pd
import plotly.express as px
import streamlit as st
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD + CACHE
# =========================
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'vehicles.csv')
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=['price', 'odometer'])

    # tratar colunas opcionais
    if 'model_year' in df.columns:
        df = df.dropna(subset=['model_year'])

    # remover outliers
    df = df[df['price'] < 100000]

    return df

df = load_data()

# =========================
# HEADER
# =========================
st.title('Vehicle Market Analytics Dashboard')

# =========================
# SIDEBAR (FILTROS)
# =========================
st.sidebar.header('Filters')

price_range = st.sidebar.slider(
    'Price Range',
    int(df['price'].min()),
    int(df['price'].max()),
    (1000, 30000)
)

if 'model_year' in df.columns:
    year_range = st.sidebar.slider(
        'Year',
        int(df['model_year'].min()),
        int(df['model_year'].max()),
        (2005, 2020)
    )
else:
    year_range = None

if 'condition' in df.columns:
    condition_options = sorted(df['condition'].dropna().unique().tolist())
    selected_conditions = st.sidebar.multiselect(
        'Condition',
        options=condition_options,
        default=condition_options
    )
else:
    selected_conditions = None

if 'fuel' in df.columns:
    fuel_options = sorted(df['fuel'].dropna().unique().tolist())
    selected_fuels = st.sidebar.multiselect(
        'Fuel',
        options=fuel_options,
        default=fuel_options
    )
else:
    selected_fuels = None

# filtro principal
filtered_df = df[
    df['price'].between(price_range[0], price_range[1])
]

if year_range:
    filtered_df = filtered_df[
        filtered_df['model_year'].between(year_range[0], year_range[1])
    ]

if selected_conditions is not None:
    filtered_df = filtered_df[filtered_df['condition'].isin(selected_conditions)]

if selected_fuels is not None:
    filtered_df = filtered_df[filtered_df['fuel'].isin(selected_fuels)]

if filtered_df.empty:
    st.warning('No listings found with the selected filters. Try broadening your filters.')
    st.stop()

# =========================
# KPIs
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Listings", len(filtered_df))
col2.metric("Avg Price", f"${int(filtered_df['price'].mean())}")
col3.metric("Avg Odometer", f"{int(filtered_df['odometer'].mean())}")

# =========================
# GRÁFICOS
# =========================

# 1. HISTOGRAMA
st.subheader('Odometer Distribution')
fig1 = px.histogram(filtered_df, x='odometer', nbins=50)
st.plotly_chart(fig1, use_container_width=True)

# 2. SCATTER
st.subheader('Price vs Odometer')
fig2 = px.scatter(
    filtered_df,
    x='odometer',
    y='price',
    color='condition' if 'condition' in df.columns else None,
    opacity=0.5
)
st.plotly_chart(fig2, use_container_width=True)

# 3. BOXPLOT
if 'condition' in df.columns:
    st.subheader('Price by Condition')
    fig3 = px.box(filtered_df, x='condition', y='price')
    st.plotly_chart(fig3, use_container_width=True)

# 4. BAR CHART
if 'model_year' in df.columns:
    st.subheader('Average Price by Year')
    price_by_year = filtered_df.groupby('model_year')['price'].mean().reset_index()

    fig4 = px.bar(price_by_year, x='model_year', y='price')
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# CLUSTERIZAÇÃO (KMEANS)
# =========================
st.subheader('Customer Segmentation (Clustering)')

cluster_df = filtered_df[['price', 'odometer']].copy()

# normalização
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_df)

# escolha de clusters
k = st.slider('Number of clusters (K)', 2, 6, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_df['cluster'] = kmeans.fit_predict(scaled_data)

# gráfico clusters
fig_cluster = px.scatter(
    cluster_df,
    x='odometer',
    y='price',
    color=cluster_df['cluster'].astype(str),
    title='Clusters: Price vs Odometer'
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =========================
# TABELAS
# =========================

# 1. TOP 10
st.subheader('Top 10 Most Expensive Vehicles')
top10 = filtered_df.sort_values(by='price', ascending=False).head(10)
st.dataframe(top10)

# 2. SUMMARY
st.subheader('Summary Statistics')
summary = filtered_df[['price', 'odometer']].describe()
st.dataframe(summary)