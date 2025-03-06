streamlit run app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt

# Set Streamlit Page Configuration
st.set_page_config(page_title="Global Terrorism Dashboard", layout="wide")

# Title
st.title("📊 Global Terrorism Analysis Dashboard")

# Upload CSV File
uploaded_file = st.file_uploader("Upload Global Terrorism CSV File", type=["csv"])

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # Display Data Overview
    st.subheader("Dataset Overview")
    st.write(df.head())

    # --- Data Cleaning (Adjust Based on Your Dataset) ---
    df["date"] = pd.to_datetime(df["date"], errors='coerce')  # Convert to DateTime
    df = df.dropna(subset=["date"])  # Drop rows with missing dates

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data")
    selected_country = st.sidebar.selectbox("Select a Country", options=df["country"].unique(), index=0)
    df_filtered = df[df["country"] == selected_country]

    # --- Attack Types Analysis ---
    st.subheader(f"Terrorist Attack Types in {selected_country}")
    attack_counts = df_filtered["attack_type"].value_counts().reset_index()
    attack_counts.columns = ["Attack Type", "Count"]
    fig_attack = px.bar(attack_counts, x="Attack Type", y="Count", title="Attack Type Distribution", color="Attack Type")
    st.plotly_chart(fig_attack)

    # --- Terrorist Groups Analysis ---
    st.subheader(f"Top Terrorist Groups in {selected_country}")
    group_counts = df_filtered["terrorist_group"].value_counts().head(10).reset_index()
    group_counts.columns = ["Terrorist Group", "Incidents"]
    fig_groups = px.bar(group_counts, x="Terrorist Group", y="Incidents", title="Top 10 Terrorist Groups", color="Terrorist Group")
    st.plotly_chart(fig_groups)

    # --- Map Visualization ---
    st.subheader("Terrorist Attack Locations")
    fig_map = px.scatter_mapbox(df_filtered, lat="latitude", lon="longitude", hover_name="city",
                                 hover_data=["attack_type", "fatalities"], zoom=2, height=500)
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map)

    # --- Time-Series Forecasting (Using Prophet) ---
    st.subheader("📈 Terrorism Trend Prediction")
    if "date" in df_filtered.columns and "fatalities" in df_filtered.columns:
        forecast_data = df_filtered[["date", "fatalities"]].groupby("date").sum().reset_index()
        forecast_data.columns = ["ds", "y"]  # Prophet expects 'ds' (date) and 'y' (value)
        
        # Train Prophet Model
        model = Prophet()
        model.fit(forecast_data)
        
        # Make Future Predictions
        future = model.make_future_dataframe(periods=365)  # Predict 1 year ahead
        forecast = model.predict(future)
        
        # Plot Predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Fatalities")
        ax.scatter(forecast_data["ds"], forecast_data["y"], color="red", label="Actual Fatalities")
        ax.legend()
        ax.set_title("Predicted vs Actual Fatalities")
        st.pyplot(fig)
    else:
        st.warning("Missing required columns for forecasting: 'date' and 'fatalities'")

else:
    st.info("Please upload a dataset to begin analysis.")
