import pandas as pd
import joblib
import os
import google.generativeai as genai

# =====================================================
# Configure Google Gemini API
# =====================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# =====================================================
# Load ML model files
# =====================================================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "stress_model.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
model_features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))

# =====================================================
# Clean column names
# =====================================================
def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_")
    )
    return df


# =====================================================
# Align dataset to model features
# =====================================================
def align_features(df):
    df = clean_columns(df)
    df = df.reindex(columns=model_features, fill_value=0)
    return df


# =====================================================
# Predict stress using ML model
# =====================================================
def predict_stress(df):
    X = align_features(df)
    preds = model.predict(X)
    labels = le.inverse_transform(preds)
    df["Predicted Stress"] = labels
    return df


# =====================================================
# Determine stress cause (rule-based)
# =====================================================
def stress_cause(row):
    causes = []

    if row.get("urea_40days", 0) < 80:
        causes.append("Low nitrogen application")

    if row.get("dap_20days", 0) < 100:
        causes.append("Low phosphorus supply")

    if row.get("potassh_50days", 0) < 40:
        causes.append("Low potassium supply")

    if row.get("30drain_in_mm", 0) > 300:
        causes.append("Excess rainfall stress")

    if not causes:
        return "Balanced crop management"

    return ", ".join(causes)


# =====================================================
# Basic fertilizer advice (fallback logic)
# =====================================================
def fertilizer_advice(row):
    advice = []

    if row.get("urea_40days", 0) < 80:
        advice.append("Apply Urea")

    if row.get("dap_20days", 0) < 100:
        advice.append("Apply DAP")

    if row.get("potassh_50days", 0) < 40:
        advice.append("Apply Potash")

    if not advice:
        return "Fertilizer application adequate"

    return ", ".join(advice)


# =====================================================
# 🤖 AI-based fertilizer recommendation (Gemini)
# =====================================================
from openai import OpenAI
import os

# Create client once
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    hf_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
else:
    hf_client = None


def ai_fertilizer_advice(row, stress_level):

    if not hf_client:
        return "AI Error: HF_TOKEN not found"

    prompt = f"""
You are an agriculture expert.

Crop: Paddy
Stress Level: {stress_level}

Field data:
- Urea at 40 days: {row.get('urea_40days', 0)} kg
- DAP at 20 days: {row.get('dap_20days', 0)} kg
- Potash at 50 days: {row.get('potassh_50days', 0)} kg
- Rainfall (30 days): {row.get('30drain_in_mm', 0)} mm

Provide:
1. Fertilizer name
2. Quantity (kg per hectare)
3. Best time to apply
4. Short explanation

Use simple farmer-friendly English.
"""

    try:
        completion = hf_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"

