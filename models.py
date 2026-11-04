import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# 1. Load the local dataset
print("Loading local Dry Bean dataset...")
df = pd.read_csv('Dry_Bean_Dataset.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Encode categorical target into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'model/label_encoder.pkl')  # Save for Streamlit inverse transform

# 3. Train-Test Split (20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# 4. Save test data (Features + Target)
test_df = X_test.copy()
test_df['Target'] = label_encoder.inverse_transform(y_test)
test_df.to_csv('test_data.csv', index=False)
print("Saved test_data.csv for Streamlit deployment.")

# 5. Preprocessing: Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')

# 6. Initialize the 5 Classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=42), ensemble=False)
}

# 7. Train, Evaluate, and Save Models
results = []

print("\n--- Training and Evaluation Results ---")
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Save the trained model
    file_path = f"model/{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, file_path)

    # Generate predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Calculate metrics (using 'weighted' for multi-class dataset)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Store results for the README comparison table
    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    print(f"Successfully trained and saved: {name}")

# 8. Display the Comparison Table
results_df = pd.DataFrame(results)
print("\nEvaluation Metrics Table :")
print(results_df.to_markdown(index=False))