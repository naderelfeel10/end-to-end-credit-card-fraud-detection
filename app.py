import streamlit as st
import pandas as pd


from src.model import load_model
from src.features import add_features

model = load_model("models/xgb_model.pkl")


st.title("📊 Fraud Detection App")
st.write("Upload a CSV file with transaction data, and the model will predict fraud risk.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV into dataframe
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # Predict button
    if st.button("🔍 Run Predictions"):
        try:
            predictions = model.predict(df)

            # Map predictions: 0 → Not Fraud, 1 → Fraud
            df["prediction"] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]

            # Style the output: Fraud = red, Not Fraud = green
            def highlight_prediction(val):
                if val == "Fraud":
                    return "color: red; font-weight: bold;"
                elif val == "Not Fraud":
                    return "color: green;"
                return ""

            styled_df = df.style.applymap(highlight_prediction, subset=["prediction"])

            st.write("### Results")
            st.dataframe(styled_df, use_container_width=True)

            # Downloadable CSV with predictions
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Error making predictions: {e}")