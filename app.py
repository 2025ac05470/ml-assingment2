from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@st.cache_resource
def load_models():
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


@st.cache_data
def load_dataset_info():
    dataset = load_breast_cancer(as_frame=True)
    return list(dataset.feature_names), list(dataset.target_names)


def calculate_metrics(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, probabilities),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1 Score": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }, predictions


st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🔬")
st.title("Breast Cancer Model Evaluation")
st.table(
    pd.DataFrame(
        [["Dheeraj Jha", "2025ac05470", "ML"]],
        columns=["Name", "BITS ID", "Sub"],
    )
)
st.write("Upload the labeled test CSV, select a trained model, and evaluate its predictions.")

try:
    models = load_models()
    feature_names, target_names = load_dataset_info()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load the required model assets: {error}")
    st.stop()

default_test_path = BASE_DIR / "test_data.csv"
if default_test_path.exists():
    st.download_button(
        "Download bundled test_data.csv",
        data=default_test_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload test_data.csv", type="csv")
selected_model = st.selectbox("Model", list(MODEL_FILES))

if uploaded_file is None:
    st.info("Upload test_data.csv to display evaluation metrics and visualizations.")
    st.stop()

test_data_source = uploaded_file

# elif default_test_path.exists():
#     test_data_source = default_test_path
# else:
#     st.info("Upload test_data.csv to display evaluation metrics and visualizations.")
#     st.stop()

try:
    test_data = pd.read_csv(test_data_source)
    missing_features = sorted(set(feature_names) - set(test_data.columns))
    if missing_features or "target" not in test_data.columns:
        missing = missing_features + ([] if "target" in test_data.columns else ["target"])
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    features = test_data[feature_names]
    target = test_data["target"].astype(int)
except (ValueError, TypeError) as error:
    st.error(f"Invalid test CSV: {error}")
    st.stop()

model = models[selected_model]
metrics, predictions = calculate_metrics(model, features, target)

st.subheader(f"Metrics: {selected_model}")
metric_columns = st.columns(6)
for column, (metric_name, value) in zip(metric_columns, metrics.items()):
    column.metric(metric_name, f"{value:.4f}")

st.subheader("Confusion Matrix")
matrix = confusion_matrix(target, predictions, labels=[0, 1])
matrix_frame = pd.DataFrame(
    matrix,
    index=[f"Actual {name.title()}" for name in target_names],
    columns=[f"Predicted {name.title()}" for name in target_names],
)
st.dataframe(matrix_frame, use_container_width=True)

st.subheader("Classification Report")
report = classification_report(
    target,
    predictions,
    labels=[0, 1],
    target_names=[name.title() for name in target_names],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

comparison_results = []
for model_name, candidate_model in models.items():
    candidate_metrics, _ = calculate_metrics(candidate_model, features, target)
    comparison_results.append({"Model": model_name, **candidate_metrics})

st.subheader("Model Comparison")
st.dataframe(
    pd.DataFrame(comparison_results).set_index("Model").round(4),
    use_container_width=True,
)
