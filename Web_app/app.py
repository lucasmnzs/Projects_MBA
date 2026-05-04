import pandas as pd
import plotly.express as px
import streamlit as st
import os

# load
csv_path = os.path.join(os.path.dirname(__file__), 'vehicles.csv')
df = pd.read_csv(csv_path)

# limpeza mínima (derivada da EDA)
df = df.dropna(subset=['odometer', 'price'])

# UI
st.header('Vehicle Listings Analysis')

# controle
show_hist = st.checkbox('Show histogram')
show_scatter = st.checkbox('Show scatter plot')

# gráficos
if show_hist:
    fig = px.histogram(df, x='odometer')
    st.plotly_chart(fig, use_container_width=True)

if show_scatter:
    fig = px.scatter(df, x='odometer', y='price')
    st.plotly_chart(fig, use_container_width=True)