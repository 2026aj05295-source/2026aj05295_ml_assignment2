import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# --- Title and Description ---
st.title("Dry Bean Classification Web App")
st.markdown("""
This application evaluates multiple machine learning models on the Dry Bean dataset. 
Please upload the `test_data.csv` file generated during the training phase to view model performance.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. Dataset Upload Option
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# 2. Model Selection Dropdown
model_options = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}
selected_model_name = st.sidebar.selectbox("Select ML Model", list(model_options.keys()))

# --- Main Application Logic ---
if uploaded_file is not None:
    try:
        # Load test data
        test_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Test Data Preview")
        st.dataframe(test_df.head())

        # Separate features and target
        # Our previous script saved the target column as 'Target'
        X_test = test_df.drop(columns=['Target'])
        y_test_labels = test_df['Target']

        # Load Preprocessors
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')

        # Apply preprocessing
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = label_encoder.transform(y_test_labels)

        # Load Selected Model
        model_filename = f"model/{model_options[selected_model_name]}"
        model = joblib.load(model_filename)

        # Make Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)

        # --- Calculate Metrics ---
        acc = accuracy_score(y_test_encoded, y_pred)
        auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test_encoded, y_pred)

        # --- Display Metrics ---
        st.write(f"### Evaluation Metrics: {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        # --- Confusion Matrix ---
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test_encoded, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

        # --- Classification Report ---
        st.write("### Classification Report")
        report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=4))

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload the `test_data.csv` file from the sidebar to proceed.")