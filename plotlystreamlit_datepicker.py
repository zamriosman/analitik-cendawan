import streamlit as st
from prophet import Prophet
import plotly.graph_objects as go
import pandas as pd
from streamlit_card import card

# Assume mushroom_df already loaded
# Ensure timestamp is datetime
mushroom_df = pd.read_csv("mushroom_dataset.csv")
mushroom_df["timestamp"] = pd.to_datetime(mushroom_df["timestamp"])

st.sidebar.info("Visualize and forecast mushroom IoT sensor data using Plotly and Prophet.")    

st.subheader("🔮 Temperature Forecast with Prophet")


# Sidebar calendar picker
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", mushroom_df["timestamp"].min().date())
with col2:
    end_date = st.date_input("End date", mushroom_df["timestamp"].max().date())

# Filter dataset by selected range
mask = (mushroom_df["timestamp"].dt.date >= start_date) & (mushroom_df["timestamp"].dt.date <= end_date)
df_filtered = mushroom_df.loc[mask, ["timestamp", "temperature_C"]].rename(
    columns={"timestamp": "ds", "temperature_C": "y"}
)

if df_filtered.empty:
    st.warning("No data available in the selected range.")
else:
    # Fit model
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_filtered)

    # Forecast horizon: extend 24h beyond selected end date
    last_date = pd.to_datetime(end_date)
    horizon_hours = 24
    future = model.make_future_dataframe(periods=horizon_hours, freq="H")
    forecast = model.predict(future)

    # Build Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered["ds"], y=df_filtered["y"],
                             mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"],
                             mode="lines", name="Upper Bound", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"],
                             mode="lines", name="Lower Bound", line=dict(dash="dash")))

    fig.update_layout(title="Temperature Forecast (Prophet)",
                      xaxis_title="Time", yaxis_title="Temperature (°C)")

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)
