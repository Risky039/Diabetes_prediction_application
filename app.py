import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# PAGE CONFIG


st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
    margin-bottom: 20px;
}

.high-risk {
    background-color: rgba(255,0,0,0.15);
    border: 1px solid rgba(255,0,0,0.4);
}

.low-risk {
    background-color: rgba(0,255,120,0.15);
    border: 1px solid rgba(0,255,120,0.4);
}

</style>
""", unsafe_allow_html=True)

# HEADER


st.title("🩺 Diabetes Prediction System")

st.write("""
This application predicts whether a patient is likely to have diabetes
using a Machine Learning model and explains the prediction using
SHAP and LIME Explainable AI techniques.
""")

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()


# PREPARE DATA

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TRAIN MODEL

@st.cache_resource
def train_model():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

model = train_model()

accuracy = model.score(X_test, y_test)

# SIDEBAR INPUTS

st.sidebar.header("Patient Information")

pregnancies = st.sidebar.slider(
    "Pregnancies",
    0,
    20,
    1
)

glucose = st.sidebar.slider(
    "Glucose",
    0,
    200,
    120
)

blood_pressure = st.sidebar.slider(
    "Blood Pressure",
    0,
    140,
    70
)

skin_thickness = st.sidebar.slider(
    "Skin Thickness",
    0,
    100,
    20
)

insulin = st.sidebar.slider(
    "Insulin",
    0,
    900,
    79
)

bmi = st.sidebar.slider(
    "BMI",
    0.0,
    70.0,
    25.0
)

diabetes_pedigree = st.sidebar.slider(
    "Diabetes Pedigree Function",
    0.0,
    3.0,
    0.5
)

age = st.sidebar.slider(
    "Age",
    1,
    100,
    33
)

# USER INPUT DATAFRAME

input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [diabetes_pedigree],
    "Age": [age]
})

# DISPLAY USER INPUT

st.subheader("📋 User Input")

st.dataframe(
    input_data,
    use_container_width=True
)


# MAKE PREDICTION

prediction = model.predict(input_data)[0]

prediction_probability = model.predict_proba(
    input_data
)[0][1]

# SHOW PREDICTION

st.subheader("🤖 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

with col2:
    st.metric(
        "Diabetes Probability",
        f"{prediction_probability:.2%}"
    )

if prediction == 1:

    st.markdown(f"""
    <div class="result-box high-risk">
        <h2>⚠️ High Diabetes Risk</h2>
        <p>
        The Machine Learning model predicts that the patient
        is likely to have diabetes.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(f"""
    <div class="result-box low-risk">
        <h2>✅ Low Diabetes Risk</h2>
        <p>
        The Machine Learning model predicts that the patient
        is unlikely to have diabetes.
        </p>
    </div>
    """, unsafe_allow_html=True)




st.header("🧠 Explainable AI")

tab1, tab2 = st.tabs([
    "SHAP Explanation",
    "LIME Explanation"
])

# SHAP EXPLANATION


with tab1:

    st.subheader("SHAP Feature Importance")

    st.write("""
    SHAP explains which features contributed most
    to the prediction.
    """)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    shap.summary_plot(
        shap_values,
        input_data,
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)


# LIME EXPLANATION

with tab2:

    st.subheader("LIME Local Explanation")

    st.write("""
    LIME explains how each feature affected
    the prediction for this specific patient.
    """)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Diabetes", "Diabetes"],
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        data_row=input_data.iloc[0],
        predict_fn=model.predict_proba
    )

    lime_fig = exp.as_pyplot_figure()

    st.pyplot(lime_fig)


# FOOTER

st.divider()

st.caption("""
Built using Streamlit, Random Forest,
SHAP, and LIME Explainable AI
""")