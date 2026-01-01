import gradio as gr
import joblib
import numpy as np

# ===============================
# Load Models & Scalers
# ===============================
lung_model = joblib.load("models/lung_patient_model.pkl")
lung_scaler = joblib.load("models/lung_patient_scaler.pkl")

heart_model = joblib.load("models/heart_patient_model.pkl")
heart_scaler = joblib.load("models/heart_patient_scaler.pkl")

breast_model = joblib.load("models/breast_patient_model.pkl")
breast_scaler = joblib.load("models/breast_patient_scaler.pkl")

# ===============================
# Helper
# ===============================
def risk_label(p):
    if p < 0.3:
        return "Low Risk"
    elif p < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# ===============================
# Prediction Functions
# ===============================
def predict_lung(
    gender, age, smoking, chronic, wheezing, coughing, sob, chest_pain
):
    X = np.array([[
        gender, age, smoking, chronic, wheezing, coughing, sob, chest_pain
    ]])
    X_scaled = lung_scaler.transform(X)
    prob = lung_model.predict_proba(X_scaled)[0][1]
    return f"{risk_label(prob)} (Probability: {prob:.2f})"

def predict_heart(age, sex, cp, exang):
    X = np.array([[age, sex, cp, exang]])
    X_scaled = heart_scaler.transform(X)
    prob = heart_model.predict_proba(X_scaled)[0][1]
    return f"{risk_label(prob)} (Probability: {prob:.2f})"

def predict_breast(age, family, pain, lump, discharge, skin):
    X = np.array([[age, family, pain, lump, discharge, skin]])
    X_scaled = breast_scaler.transform(X)
    prob = breast_model.predict_proba(X_scaled)[0][1]
    return f"{risk_label(prob)} (Probability: {prob:.2f})"

# ===============================
# Gradio UI
# ===============================
with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown("## 🏥 Patient Health Risk Screening Tool")
    gr.Markdown("_Risk screening only. Not a medical diagnosis._")

    # ---------------- Lung Cancer ----------------
    with gr.Tab("🫁 Lung Cancer"):
        gr.Markdown("**Gender** — 0 = Female, 1 = Male")
        gender = gr.Radio([0, 1])

        age = gr.Number(label="Age")

        gr.Markdown("**Smoking** — 0 = No, 1 = Yes")
        smoking = gr.Radio([0, 1])

        gr.Markdown("**Chronic Disease** — 0 = No, 1 = Yes")
        chronic = gr.Radio([0, 1])

 #       gr.Markdown("**Fatigue** — 0 = No, 1 = Yes")
  #      fatigue = gr.Radio([0, 1])

        gr.Markdown("**Wheezing** — 0 = No, 1 = Yes")
        wheezing = gr.Radio([0, 1])

        gr.Markdown("**Coughing** — 0 = No, 1 = Yes")
        coughing = gr.Radio([0, 1])

        gr.Markdown("**Shortness of Breath** — 0 = No, 1 = Yes")
        sob = gr.Radio([0, 1])

        gr.Markdown("**Chest Pain** — 0 = No, 1 = Yes")
        chest_pain = gr.Radio([0, 1])

        lung_out = gr.Textbox(label="Risk Result")
        gr.Button("Check Lung Cancer Risk").click(
            predict_lung,
            [gender, age, smoking, chronic, wheezing, coughing, sob, chest_pain],
            lung_out
        )

    # ---------------- Heart Disease ----------------
    with gr.Tab("❤️ Heart Disease"):
        age_h = gr.Number(label="Age")

        gr.Markdown("**Sex** — 0 = Female, 1 = Male")
        sex = gr.Radio([0, 1])

        gr.Markdown(
            "**Chest Pain Severity** — "
            "0 = No pain, 1 = Mild, 2 = Moderate, 3 = Severe"
        )
        cp = gr.Radio([0, 1, 2, 3])

        gr.Markdown("**Chest Pain During Exercise** — 0 = No, 1 = Yes")
        exang = gr.Radio([0, 1])

        heart_out = gr.Textbox(label="Risk Result")
        gr.Button("Check Heart Disease Risk").click(
            predict_heart,
            [age_h, sex, cp, exang],
            heart_out
        )

    # ---------------- Breast Cancer ----------------
    with gr.Tab("🎗️ Breast Cancer Screening"):
        age_b = gr.Number(label="Age")

        gr.Markdown("**Family History** — 0 = No, 1 = Yes")
        family = gr.Radio([0, 1])

        gr.Markdown("**Breast Pain** — 0 = No, 1 = Yes")
        pain = gr.Radio([0, 1])

        gr.Markdown("**Lump Felt** — 0 = No, 1 = Yes")
        lump = gr.Radio([0, 1])

        gr.Markdown("**Nipple Discharge** — 0 = No, 1 = Yes")
        discharge = gr.Radio([0, 1])

        gr.Markdown("**Skin Changes** — 0 = No, 1 = Yes")
        skin = gr.Radio([0, 1])

        breast_out = gr.Textbox(label="Risk Result")
        gr.Button("Check Breast Cancer Risk").click(
            predict_breast,
            [age_b, family, pain, lump, discharge, skin],
            breast_out
        )

app.launch()
