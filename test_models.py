import joblib
import numpy as np
import sklearn
import os
import traceback
import warnings

warnings.filterwarnings('ignore')

files = [ "logistic_regression.pkl", "decision_tree.pkl", "knn.pkl", "gaussian_nb.pkl", "svm.pkl", "scaler.pkl", "feature_names.pkl" ]

for file in files:
    path = os.path.join("model", file)
    print(f"\nTesting: {file}")
    try:
        obj = joblib.load(path)
        print("SUCCESS:", type(obj))
    except Exception:
        print("FAILED:")
