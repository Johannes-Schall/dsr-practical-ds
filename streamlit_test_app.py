import streamlit as st
import pandas as pd
import numpy as np
from plotly import graph_objs as go
import plotly.express as px

st.title("My first app")
a = st.sidebar.radio("Select one:", [1, 2])
if a == 1:
    st.video("https://www.youtube.com/watch?v=NViKsiGP4TY")
else:
    st.write("You selected 2")

col1, col2 = st.columns(2)
col1.write("This is column 1")
col2.write("This is column 2")

st.checkbox("Show dataframe")
st.multiselect("Buy", ["Apple", "Banana", "Orange"])
num_rows = st.slider("Select a number of rows", 1, 100)
st.time_input("Set an alarm for")


@st.cache_data
def load_data(
    data_url="https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz",
    nrows=1000,
):
    return pd.read_csv(data_url, nrows=nrows)


data = load_data()
data.rename(columns={"Lat": "LAT", "Lon": "LON"}, inplace=True)
st.dataframe(data.head(num_rows))

st.write("Map of pickups (first 100 max)")
st.map(data.loc[:num_rows, ["LAT", "LON"]])

# Change the date column to datetime
data["Date/Time"] = pd.to_datetime(data["Date/Time"])

# Histogram of counts by hour with plotly express
hist_values = np.histogram(data["Date/Time"].dt.hour, bins=24, range=(0, 24))[0]
st.write("Histogram of pickups by hour")
# plot using plotly express
fig = px.bar(x=range(24), y=hist_values, labels={"x": "Hour", "y": "Count"})
st.plotly_chart(fig)
