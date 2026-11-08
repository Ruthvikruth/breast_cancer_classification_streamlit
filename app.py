import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

# Model directory
MODEL_DIR = "model"
st.set_page_config( page_title = "Breast Cancer Classification", page_icon = "🩺", layout = "wide" )


# Load saved models
@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load( os.path.join(MODEL_DIR, "logistic_regression.pkl") )
        , "Decision Tree": joblib.load( os.path.join(MODEL_DIR, "decision_tree.pkl") )
        , "KNN": joblib.load( os.path.join(MODEL_DIR, "knn.pkl") )
        , "Gaussian Naive Bayes": joblib.load( os.path.join(MODEL_DIR, "gaussian_nb.pkl") )
        , "SVM": joblib.load( os.path.join(MODEL_DIR, "svm.pkl") )
    }

    scaler = joblib.load( os.path.join(MODEL_DIR, "scaler.pkl") )
    feature_names = joblib.load( os.path.join(MODEL_DIR, "feature_names.pkl") )

    return models, scaler, feature_names

models, scaler, feature_names = load_models()

st.title("Breast Cancer Classification")
st.write(
    "Compare five machine learning classifiers "
    "using the Wisconsin Breast Cancer dataset."
)

st.success("All models loaded successfully!")
st.divider()
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader( "Upload your test CSV file", type = ["csv"] )


if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")
    st.subheader("Dataset Preview:")
    st.dataframe(test_df.head())
    st.write("Dataset Dimensions:", test_df.shape)

    if "diagnosis" not in test_df.columns:
        st.error("The uploaded CSV must contain a 'diagnosis' column.")

    else:
        y_test = test_df["diagnosis"]

        if y_test.dtype == "object":
            y_test = y_test.astype(str).str.strip().str.upper()
            if set( y_test.unique() ).issubset( {"M", "B"} ):
                y_test = y_test.map( {"M": 1, "B": 0} )
            else:
                y_test = pd.to_numeric(y_test, errors = "coerce")

        else:
            y_test = pd.to_numeric(y_test, errors = "coerce")

        if y_test.isna().any() or not y_test.isin([0, 1]).all():
            st.error( "Invalid diagnosis values. Use 0 (Benign)"
                "or 1 (Malignant), or B/M labels." )

        else:
            missing_features = [ col for col in feature_names if col not in test_df.columns ]

            if missing_features:
                st.error(f"Missing features: {missing_features}")

            else:
                X_test = test_df[feature_names].copy()
                X_test = X_test.apply( pd.to_numeric, errors = "coerce" )

                if X_test.isna().any().any() or not np.isfinite( X_test.to_numpy() ).all():
                    st.error(
                        "The dataset contains missing or invalid "
                        "feature values."
                    )

                else:
                    st.write("Number of test records:", len(X_test))
                    st.write("Number of features:", X_test.shape[1])

                st.write("Number of features:", X_test.shape[1] )

                X_test_scaled = scaler.transform( X_test )
                predictions = {}

                for model_name, model in models.items():
                    if model_name in [ "Logistic Regression", "KNN", "SVM" ]:
                        predictions[model_name] = model.predict(X_test_scaled)
                    else:
                        predictions[model_name] = model.predict(X_test)

                model_results = {}
                for model_name, model in models.items():
                    model_predictions = predictions[model_name]
                    if model_name in [ "Logistic Regression", "KNN", "SVM" ]:
                        model_input = X_test_scaled
                    else:
                        model_input = X_test

                    model_probabilities = model.predict_proba(model_input)[:, 1]
                    model_results[model_name] = {
                        "Benign Predictions": int(np.sum(model_predictions == 0)),
                        "Malignant Predictions": int(np.sum(model_predictions == 1)),
                        "Accuracy": accuracy_score(y_test, model_predictions),
                        "AUC Score": roc_auc_score(y_test, model_probabilities),
                        "Precision": precision_score( y_test, model_predictions, zero_division = 0 ),
                        "Recall": recall_score( y_test, model_predictions, zero_division = 0 ),
                        "F1 Score": f1_score( y_test, model_predictions, zero_division = 0 ),
                        "MCC": matthews_corrcoef(y_test, model_predictions)
                    }

                results_df = pd.DataFrame(model_results)
                formatted_results = results_df.copy()

                metric_rows = [ "Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC" ]

                formatted_results.loc[metric_rows] = ( formatted_results.loc[metric_rows].round(4) )

                st.divider()
                st.subheader("Overall Model Performance Comparison")
                st.dataframe( formatted_results, use_container_width = True )
                st.divider()

                st.header("Detailed Model Evaluation")
                selected_model_name = st.selectbox( "Select a classification model", list(models.keys()) )
                selected_predictions = predictions[selected_model_name]

                st.subheader(f"{selected_model_name} - Confusion Matrix")

                cm = confusion_matrix( y_test, selected_predictions, labels=[0, 1] )
                cm_df = pd.DataFrame( cm, index = ["Actual Benign", "Actual Malignant"], columns = ["Predicted Benign", "Predicted Malignant"] )
                st.dataframe(cm_df, use_container_width = True)

                st.subheader("Classification Report")
                report = classification_report( y_test, selected_predictions, labels = [0, 1], target_names = ["Benign", "Malignant"], output_dict = True, zero_division = 0 )
                report_df = pd.DataFrame(report).transpose()

                st.dataframe( report_df.round(4), use_container_width = True )
