import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# Load and prepare data
@st.cache_data
def load_data():
    return pd.read_csv("final_df.csv")


@st.cache_resource
def train_models():
    data = load_data()

    # Convert float columns (excluding age, bmi) to int
    exclude_columns = ['age', 'body_mass_index']
    float_cols = [col for col in data.select_dtypes(include='float64').columns if col not in exclude_columns]
    data[float_cols] = data[float_cols].astype(int)

    df_1 = data[:80000].copy()
    df_2 = data[80000:].copy()

    data_0 = df_1[df_1["heart_attack_or_stroke_occurred"] == 0]
    data_1 = df_1[df_1["heart_attack_or_stroke_occurred"] == 1]
    df_class_1_over = data_1.sample(len(data_0), replace=True)
    df_over = pd.concat([data_0, df_class_1_over], axis=0)

    X_train = df_over.drop(columns=['heart_attack_or_stroke_occurred'])
    y_train = df_over['heart_attack_or_stroke_occurred']

    A = df_2.drop(columns=['heart_attack_or_stroke_occurred'])
    b = df_2['heart_attack_or_stroke_occurred']
    X_valid, _, y_valid, _ = train_test_split(A, b, test_size=0.5, random_state=42, stratify=b)

    model_scores = {}

    # Logistic Regression
    log_model = LogisticRegression(max_iter=100000)
    log_model.fit(X_train, y_train)
    acc_log = accuracy_score(y_valid, log_model.predict(X_valid))
    model_scores["Logistic Regression"] = acc_log

    return log_model, list(X_train.columns), model_scores


def main():
    st.title("Heart Attack and Stroke Prediction")
    st.markdown("Step-by-step feature input using the best model.")

    model, feature_names, model_scores = train_models()


    # Feature Descriptions
    feature_descriptions = {
        "gender": "The gender of the patient.  Female or Male",
        "age": "The age of the patient in years",
        "body_mass_index": "The body mass index of the individual",
        "smoker": "Does the person have a history of smoking? (Yes or No)",
        "systolic_blood_pressure": "Systolic blood pressure in mmHg",
        "hypertension_treated": "Is the person currently on hypertension treatment? (Yes or No)",
        "family_history_of_cardiovascular_disease": "Any family history of cardiovascular disease? (Yes or No)",
        "atrial_fibrillation": "Does the person have atrial fibrillation? (Yes or No)",
        "chronic_kidney_disease": "Presence of chronic kidney disease? (Yes or No)",
        "rheumatoid_arthritis": "Presence of rheumatoid arthritis? (Yes or No)",
        "diabetes": "Presence of diabetes? (Yes or No )",
        "chronic_obstructive_pulmonary_disorder": "Presence of chronic obstructive pulmonary disorder? (Yes or No)",
        "forced_expiratory_volume": "The FEV1: air exhaled in 1 second as % of predicted FEV1",
        "time": "The time to event, or time to censoring, in years (e.g 1,2,3,4, ...)"
    }

    binary_yes_no_fields = [
        'smoker', 'hypertension_treated', 'family_history_of_cardiovascular_disease',
        'atrial_fibrillation', 'chronic_kidney_disease', 'rheumatoid_arthritis',
        'diabetes', 'chronic_obstructive_pulmonary_disorder'
    ]

    # Initialize session state
    if 'feature_index' not in st.session_state:
        st.session_state.feature_index = 0
        st.session_state.user_inputs = {}

    # Done collecting input
    if st.session_state.feature_index >= len(feature_names):
        st.subheader("✅ All inputs collected. Ready to Predict!")

        input_data = [st.session_state.user_inputs.get(feature, 0) for feature in feature_names]
        user_input_df = pd.DataFrame([input_data], columns=feature_names)

        st.write("### Your Input Data")
        st.dataframe(user_input_df)

        if st.button("Predict"):
            prediction = model.predict(user_input_df)[0]
            probability = model.predict_proba(user_input_df)[0][1]

            st.subheader("Prediction Result")
            st.success("⚠️ Risk Detected!" if prediction == 1 else "✅ No Immediate Risk Detected.")
            st.write(f"**Probability of Heart Attack or Stroke:** {probability:.2%}")

        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        return

    # Input current feature
    current_feature = feature_names[st.session_state.feature_index]
    st.subheader(f"Input: {current_feature.replace('_', ' ').capitalize()}")

    description = feature_descriptions.get(current_feature, "Please provide the value.")
    st.info(description)

    # Input logic
    if current_feature.lower() == "gender":
        value = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    elif current_feature in binary_yes_no_fields:
        value = st.selectbox(current_feature.replace("_", " ").capitalize(), options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    elif current_feature in ["age", "body_mass_index", "forced_expiratory_volume", "time"]:
        value = st.number_input(current_feature, min_value=0.0, step=0.1, format="%.1f")
    else:
        value = st.number_input(current_feature, min_value=0, step=1)

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous") and st.session_state.feature_index > 0:
            st.session_state.feature_index -= 1

    with col2:
        if st.button("Next"):
            st.session_state.user_inputs[current_feature] = value
            st.session_state.feature_index += 1
            st.rerun()



if __name__ == "__main__":
    main()

