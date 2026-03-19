import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Medical Cost Prediction", layout="wide")

# ===============================
# Dark Mode Toggle
# ===============================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle = st.toggle("🌗 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = toggle

# ===============================
# Theme
# ===============================
if st.session_state.dark_mode:
    bg = "linear-gradient(135deg, #0f172a, #1e293b)"
    text = "#f1f5f9"
    glass = "rgba(255,255,255,0.05)"
else:
    bg = "linear-gradient(135deg, #e0ecff, #f8fbff)"
    text = "#0f172a"
    glass = "rgba(255,255,255,0.6)"

# ===============================
# CSS (🔥 UPDATED)
# ===============================
st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
}}

.creator {{
    position: absolute;
    top: 10px;
    left: 20px;
    font-weight: 700;
}}

.title {{
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-top: 30px;
}}

.glass {{
    background: {glass};
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 25px;
}}

.panel-title {{
    text-align: center;
    font-size: 22px;
    font-weight: 700;
}}

.prediction {{
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: white;
    height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.accuracy {{
    font-size: 14px;
    margin-top: 8px;
}}

/* 🔥 BUTTON STYLE */
.stButton>button {{
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    margin: 0px !important;
    transition: all 0.3s ease;
}}

/* 🔥 HOVER GLOW EFFECT */
.stButton>button:hover {{
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(102, 126, 234, 0.7);
}}

/* 🔥 REMOVE COLUMN GAP COMPLETELY */
div[data-testid="column"] {{
    padding-left: 0px !important;
    padding-right: 0px !important;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# PDF Generator
# ===============================
def generate_pdf(age, bmi, children, sex, smoker, region, prediction, lower, upper):
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Medical Insurance Prediction Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Generated on: {datetime.datetime.now()}", styles['Normal']))

    for label, value in [
        ("Age", age), ("BMI", bmi), ("Children", children),
        ("Sex", sex), ("Smoker", smoker), ("Region", region)
    ]:
        content.append(Paragraph(f"<b>{label}:</b> {value}", styles['Normal']))

    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>Estimated Cost:</b> ₹ {prediction:,.2f}", styles['Heading2']))
    content.append(Paragraph(f"Range: ₹ {lower:,.2f} - ₹ {upper:,.2f}", styles['Normal']))

    doc.build(content)

    with open("report.pdf", "rb") as f:
        return f.read()

# ===============================
# Header
# ===============================
st.markdown('<div class="creator">Created by Archit Tomar</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ===============================
# Layout
# ===============================
left_col, right_col = st.columns([1, 2])

# ===============================
# LEFT PANEL
# ===============================
with left_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="panel-title">📊 Model Information</div>', unsafe_allow_html=True)

    st.markdown("""
    Model Used: Random Forest Regressor  
    Model R² Score: 0.69
    """)

    st.markdown("""
    Features:
    - Age  
    - BMI  
    - Children  
    - Sex  
    - Smoker  
    - Region  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# RIGHT PANEL
# ===============================
with right_col:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown('<div class="panel-title">🧾 Enter Patient Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 25)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        children = st.slider("Children", 0, 5, 0)

    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"])

    st.markdown("<br>", unsafe_allow_html=True)

    # ✅ BUTTONS (NO GAP + GLOW)
    btn1, btn2 = st.columns([1, 1], gap="small")

    with btn1:
        predict_btn = st.button("🚀 Predict Insurance Cost")

    with btn2:
        download_container = st.container()

    if predict_btn:
        with st.spinner("🔍 Analyzing patient profile..."):
            time.sleep(1)

        input_data = pd.DataFrame({
            "age": [age],
            "bmi": [bmi],
            "children": [children],
            "sex_male": [1 if sex == "male" else 0],
            "smoker_yes": [1 if smoker == "yes" else 0],
            "region_northwest": [1 if region == "northwest" else 0],
            "region_southeast": [1 if region == "southeast" else 0],
            "region_southwest": [1 if region == "southwest" else 0]
        })

        input_data[["age", "bmi", "children"]] = scaler.transform(
            input_data[["age", "bmi", "children"]]
        )

        log_pred = model.predict(input_data)
        final_pred = np.expm1(log_pred)

        std_dev = final_pred[0] * 0.15
        lower = final_pred[0] - std_dev
        upper = final_pred[0] + std_dev

        # ✅ Download button beside predict
        with download_container:
            pdf = generate_pdf(age, bmi, children, sex, smoker, region, final_pred[0], lower, upper)

            st.download_button(
                "📄 Download Report",
                data=pdf,
                file_name="insurance_prediction_report.pdf",
                mime="application/pdf"
            )

        # Output
        out1, out2 = st.columns([1, 1], gap="large")

        with out1:
            st.markdown(f"""
            <div class="prediction">
                Estimated Insurance Cost <br><br>
                ₹ {final_pred[0]:,.2f}
                <div class="accuracy">
                    Range: ₹ {lower:,.2f} - ₹ {upper:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with out2:
            feature_names = [
                "age", "bmi", "children",
                "sex_male", "smoker_yes",
                "region_northwest", "region_southeast", "region_southwest"
            ]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.set_title("📈 Feature Importance")

            st.pyplot(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)