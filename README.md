<h3>Problem Statement: <h4>The primary challenge in early breast cancer detection is accurately and rapidly classifying breast mass tumors as either Malignant (cancerous) or Benign (non-cancerous) based on features computed from digitized FNA images.</h4> </h3>

<h3>Dataset Description: <h4>The Breast Cancer Wisconsin (Diagnostic) Dataset on Kaggle is a binary classification dataset used for machine learning and medical data analysis.</h4> </h3>

<h3>Github Repository Link: <h4>https://github.com/Ruthvikruth/breast_cancer_classification_streamlit.git</h4> </h3>

<br/>
<h3>Model Performance:</h3>

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

<br/>
<h3>Performance Observations:</h3>

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression |This is the second best model that could be chosen after SVM due to its high recall and precision.|
| Decision Tree |Although not the best model but the precision and recall scores are high and good enough to be used!|
| KNN |Fairly good recall score and high precision value.|
| Naive Bayes |Lower recall score compared to other models performance although precision is still high.|
| Support Vector Machine | The best model that works with this dataset as it has the highest recall and precision scores. |
| **Overall Winner for Your Dataset** | **Support Vector Machine would be the winner in this case. For a cancer patient detection use case, the recall and precision are the metrics of highest priority. Since SVM has the highest recall and precision among the 5 models, we can declare it as the winner!** |

