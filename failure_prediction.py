import joblib
import numpy as np
import pandas as pd
import streamlit as st


model= joblib.load('model.pkl')

def preprocess(df, training=True):

    df = df.copy()

    df["Temperature_diff"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    omega = 2 * np.pi * df["Rotational speed [rpm]"] / 60
    df["power"] = omega * df["Torque [Nm]"]

    df["Wear_torque"] = (
            df["Tool wear [min]"] * df["Torque [Nm]"]
    )

    df["temperature_ratio"] = (
            df["Process temperature [K]"] / df["Air temperature [K]"]
    )

    df["torque_speed_ratio"] = (
            df["Torque [Nm]"] / df["Rotational speed [rpm]"]
    )

    df["wear_power_interaction"] = (
            df["Tool wear [min]"] * df["power"]
    )

    return df

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙",
    layout="wide"
)

st.title("⚙ Machine Failure Prediction System")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read File
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data☹️")

    st.dataframe(df.head())

    # Feature Engineering
    new_df = preprocess(df)

    st.subheader("Feature Engineered Data🤖")

    st.dataframe(new_df.head())

    # Prediction
    prediction = model.predict(new_df)

    new_df["Prediction"] = prediction

    st.success("Prediction Completed😍")

    st.dataframe(new_df)

    csv = new_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction",
        csv,
        "prediction.csv",
        "text/csv"
    )


