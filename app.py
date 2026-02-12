import streamlit as st
import pandas as pd
from utils import (
    predict_stress,
    stress_cause,
    fertilizer_advice,
    ai_fertilizer_advice
)

st.set_page_config(page_title="Paddy Stress Prediction", layout="wide")

st.title("🌾 Paddy Crop Stress Prediction System")
st.markdown("AI-powered system to detect crop stress and provide fertilizer recommendations.")

# Sidebar
mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Entry", "Upload CSV"]
)

# =====================================================
# Manual Entry Mode
# =====================================================
if mode == "Manual Entry":
    st.header("Farm Input")

    col1, col2 = st.columns(2)

    with col1:
        hectares = st.number_input("Farm Area (ha)", 1, 10, 3)
        seedrate = st.number_input("Seed Rate (kg)", 50, 200, 100)
        dap = st.number_input("DAP Applied (kg)", 0, 300, 150)

    with col2:
        urea = st.number_input("Urea Applied (kg)", 0, 300, 100)
        potash = st.number_input("Potash Applied (kg)", 0, 200, 40)
        rainfall = st.number_input("Rainfall First 30 Days (mm)", 0, 500, 120)

    if st.button("Predict Stress"):

        input_data = pd.DataFrame([{
            "hectares": hectares,
            "seedratein_kg": seedrate,
            "dap_20days": dap,
            "urea_40days": urea,
            "potassh_50days": potash,
            "30drain_in_mm": rainfall
        }])

        result = predict_stress(input_data)
        stress = result["Predicted Stress"].iloc[0]

        cause = stress_cause(input_data.iloc[0])

        st.subheader("Prediction Result")

        # Stress Level
        if stress == "High":
            st.error(f"Stress Level: {stress}")
        elif stress == "Medium":
            st.warning(f"Stress Level: {stress}")
        else:
            st.success(f"Stress Level: {stress}")

        # Cause
        st.info(f"Cause: {cause}")

        # 🌱 Fertilizer Recommendation (AI)
        st.markdown("### 🌱 Fertilizer Recommendation")
        ai_advice = ai_fertilizer_advice(input_data.iloc[0], stress)
        st.write(ai_advice)


# =====================================================
# CSV Upload Mode
# =====================================================
elif mode == "Upload CSV":
    st.header("Bulk Farm Prediction & Analytics")

    uploaded_file = st.file_uploader("Upload Farm Dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Prediction
        result = predict_stress(df)

        result["Cause"] = result.apply(stress_cause, axis=1)
        result["Rule_Based_Advice"] = result.apply(fertilizer_advice, axis=1)
        result["AI_Fertilizer_Advice"] = result.apply(
            lambda row: ai_fertilizer_advice(row, row["Predicted Stress"]),
            axis=1
        )

        st.subheader("Prediction Results")
        st.dataframe(result, use_container_width=True)

        # Stress Summary
        st.subheader("Stress Summary")

        counts = result["Predicted Stress"].value_counts()

        c1, c2, c3 = st.columns(3)
        c1.metric("High Stress Farms", counts.get("High", 0))
        c2.metric("Medium Stress Farms", counts.get("Medium", 0))
        c3.metric("Low Stress Farms", counts.get("Low", 0))

        # Analytics
        st.subheader("📊 Analytics Dashboard")

        st.write("Stress Distribution")
        st.bar_chart(result["Predicted Stress"].value_counts())

        # Download
        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )
