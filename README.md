# 🩺 Type 2 Diabetes Prediction System

An end-to-end, beginner-friendly Machine Learning project and interactive web application built with **Python**, **Scikit-Learn**, **Pandas**, and **Streamlit**.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Details & Limitations](#-dataset-details--limitations)
3. [Machine Learning Pipeline & Methodology](#-machine-learning-pipeline--methodology)
4. [File Structure & Major Components](#-file-structure--major-components)
5. [Installation & Setup](#-installation--setup)
6. [Training the Models](#-training-the-models)
7. [Running the Automated Tests](#-running-the-automated-tests)
8. [Launching the Web Application](#-launching-the-web-application)
9. [Troubleshooting Common Issues](#-troubleshooting-common-issues)
10. [Safety, Ethical & Clinical Disclaimers](#-safety-ethical--clinical-disclaimers)

---

## 🎯 Project Overview

This project provides a complete, working machine learning application—not just an isolated notebook. It guides you from data ingestion and cleaning through leak-free cross-validation to an interactive web dashboard where users can explore data, compare models, and test clinical inputs.

### Key Goals:
- **Beginner-Friendly**: Clear code structure, extensive inline documentation, and friendly UI tooltips.
- **Leak-Free ML Engineering**: Strict separation between training and test sets using Scikit-Learn `Pipeline` objects.
- **Rigor in Model Selection**: Comparing models via **5-Fold Stratified Cross-Validation on the training set**, avoiding optimistic test-set data snooping.
- **Comprehensive Evaluation**: Metrics including Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrices.
- **Interactive UI**: A multi-page Streamlit dashboard featuring exploratory data analysis, visual diagnostic curves, and interactive risk prediction.

---

## 🔬 Dataset Details & Limitations

### 1. Source & Reference
- **Dataset**: *Pima Indians Diabetes Database* (PIDD)
- **Origin**: Gathered by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/34/pima+indians+diabetes).
- **Primary Publication**: Smith, C.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. In Proceedings of the Annual Symposium on Computer Application in Medical Care (pp. 261–265).

### 2. Population
- All 768 participants are **adult females aged 21 and older of Pima Indian heritage**, residing near Phoenix, Arizona, USA.

### 3. Target Variable
- **`Outcome`**: Binary classification target.
  - `0` (Negative): Tested negative for diabetes onset.
  - `1` (Positive): Tested positive for diabetes onset within 5 years of examination according to World Health Organization (WHO) criteria.
- **Class Distribution**: 500 Non-diabetic (65.1%) vs 268 Diabetic (34.9%), demonstrating moderate class imbalance.

### 4. Features & The "Invalid Zeros" Problem
The dataset contains 8 clinical features:
1. `Pregnancies`: Number of times pregnant (values of `0` are valid).
2. `Glucose`: 2-hour plasma glucose concentration in an oral glucose tolerance test (mg/dL).
3. `BloodPressure`: Diastolic blood pressure (mm Hg).
4. `SkinThickness`: Triceps skinfold thickness (mm).
5. `Insulin`: 2-Hour serum insulin (μU/mL).
6. `BMI`: Body Mass Index (weight in kg / $(\text{height in m})^2$).
7. `DiabetesPedigreeFunction`: Genetic family risk score.
8. `Age`: Age in years (range 21–81).

> [!IMPORTANT]
> **Data Cleaning Discovery**: In medical measurements, a living person cannot have a Glucose of 0, Blood Pressure of 0, or BMI of 0. In this dataset, zeros in `Glucose` (5), `BloodPressure` (35), `SkinThickness` (227), `Insulin` (374), and `BMI` (11) represent **missing data**, not actual physiological zeros!
> Our pipeline automatically flags these values as `NaN` and handles them using median imputation inside Scikit-Learn pipelines.

### 5. Documented Limitations & Bias
- **Demographic Bias**: The findings cannot be directly applied to men, children, or non-Pima ethnic groups without external validation.
- **Unrecorded Diagnostics**: Modern key diagnostic indicators (e.g., HbA1c) were not measured in this historical dataset.
- **Sample Size**: 768 samples is relatively small, requiring regularization and cross-validation to prevent overfitting.

---

## 🧠 Machine Learning Pipeline & Methodology

```
Raw CSV (768 records)
        │
        ▼
Data Auditing (Replace invalid 0s with NaN for clinical features)
        │
        ▼
Stratified Train/Test Split (80% Train [614], 20% Test [154])
        │
        ├───────────────────────────────────────────┐
        ▼                                           ▼
[Training Set (80%)]                         [Held-Out Test Set (20%)]
        │                                           │
        ▼                                           │
5-Fold Stratified Cross-Validation                  │
  ├── 1. Median Imputation                          │
  ├── 2. Feature Scaling (for Linear Model)         │
  └── 3. Estimator (LogReg / DT / RF)               │
        │                                           │
        ▼                                           │
Model Comparison & Selection (Best CV F1/AUC)       │
        │                                           │
        ▼                                           │
Final Fit on Full Training Set ─────────────────────┘
        │
        ▼
Unbiased Test Set Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
        │
        ▼
Artifact Serialization (Pipeline, Models, Metrics -> /models/)
```

### Models Evaluated:
1. **Logistic Regression (Baseline)**:
   - Linear probabilistic classifier with L2 regularization.
   - Preprocessing: Median Imputation + `StandardScaler`.
2. **Decision Tree Classifier**:
   - Non-linear interpretable tree with `max_depth=4` and `min_samples_leaf=10` to prevent tree memorization on small data.
3. **Random Forest Classifier**:
   - Ensemble of 100 bagging trees with restricted depth and leaf constraints.

### Why Avoid Selecting Models on Test-Set Alone?
Evaluating models repeatedly on the test set and selecting whichever scores highest is known as **test-set data snooping**. It overfits your model choices to that specific test subset. Instead, we use **Stratified 5-Fold Cross-Validation on the training set** to pick the model that generalizes across folds, keeping the test set purely for final confirmation.

---

## 📁 File Structure & Major Components

```
diabetes-ml-project/
│
├── data/
│   └── diabetes.csv           # Publicly documented Pima Indians Diabetes Dataset
│
├── models/
│   ├── best_model.joblib      # Serialized winning pipeline (imputer + scaler + model)
│   ├── all_models.joblib      # Dictionary with all 3 trained pipelines, metadata & curve data
│   └── metrics.json           # JSON summary of cross-validation & test metrics
│
├── tests/
│   ├── test_pipeline.py       # Unit tests for preprocessing, zero-replacement, and pipelines
│   └── test_app.py            # Integration tests for app loading and model inference
│
├── app.py                     # Streamlit web application (4 interactive pages)
├── train_model.py             # Script to load data, run CV, train pipelines, and save models
├── requirements.txt           # Python dependencies with minimum version bounds
└── README.md                  # Comprehensive project documentation
```

### Purpose of Every Major File:
- **`train_model.py`**: The training engine. It handles missing data conversions, trains Scikit-Learn pipelines, conducts 5-fold cross-validation, benchmarks models, and persists joblib artifacts into `models/`.
- **`app.py`**: The front-end user interface. Provides multi-page navigation for the project overview, interactive data visualizations, model diagnostics (ROC, PR, confusion matrix), and an interactive clinical risk screener.
- **`requirements.txt`**: Specifies all required Python libraries (`pandas`, `numpy`, `scikit-learn`, `streamlit`, `matplotlib`, `seaborn`, `joblib`, `pytest`).
- **`tests/test_pipeline.py` & `test_app.py`**: Pytest test suites verifying dataset integrity, zero imputation, pipeline inference, probability validity, and edge-case handling.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13, or 3.14 recommended)
- `pip` package manager

### Step 1: Open Terminal / PowerShell
Navigate to the project root directory:
```bash
cd "c:\Users\Student\Documents\diabetes-ml-project"
```

### Step 2 (Optional but Recommended): Create a Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
*(If you have Anaconda or multiple Python versions installed, invoke the specific Python binary, e.g.: `C:\ProgramData\anaconda3\python.exe -m pip install -r requirements.txt`)*

---

## ⚙️ Training the Models

To execute data validation, 5-fold cross-validation, and pipeline artifact generation:

```bash
python train_model.py
```

### Expected Output:
```text
[1] Dataset loaded successfully from data\diabetes.csv.
    - Total records: 768, Features: 8
    - Identified invalid zeros treated as missing values:
      * Glucose: 5 zeros (0.7%)
      * BloodPressure: 35 zeros (4.6%)
      * SkinThickness: 227 zeros (29.6%)
      * Insulin: 374 zeros (48.7%)
      * BMI: 11 zeros (1.4%)

Training set size: 614 samples, Test set size: 154 samples

[2] Performing 5-Fold Stratified Cross-Validation on Training Set...
---------------------------------------------------------------------------
Model                  | Accuracy   | Precision  | Recall     | F1 Score   | ROC-AUC   
---------------------------------------------------------------------------
Logistic Regression    | 0.788±0.02 | 0.764±0.06 | 0.575±0.02 | 0.655±0.02 | 0.843±0.02
Decision Tree          | 0.725±0.05 | 0.638±0.10 | 0.542±0.04 | 0.581±0.04 | 0.778±0.05
Random Forest          | 0.764±0.01 | 0.723±0.05 | 0.533±0.04 | 0.611±0.02 | 0.838±0.02
---------------------------------------------------------------------------
Selected best model via Training Cross-Validation: 'Logistic Regression' (Mean CV F1: 0.655)

[3] Evaluating Models on Held-Out Test Set (20% of data)...
...
[4] Saved best pipeline ('Logistic Regression') to 'models\best_model.joblib'.
```

---

## 🧪 Running the Automated Tests

Run the test suite with `pytest`:

```bash
python -m pytest tests/ -v
```

### All 10 Tests Cover:
- Dataset existence, column integrity, and class distribution
- Correct replacement of biologically invalid zeros with `NaN`
- Preservation of valid zeros in `Pregnancies`
- Pipeline median imputation and inference
- Probability validity (between 0.0 and 1.0, summing to 1.0)
- Model artifact loading and metrics JSON integrity
- Batch inference and extreme clinical input handling

---

## 🌐 Launching the Web Application

Launch the Streamlit dashboard in your local browser:

```bash
python -m streamlit run app.py
```

Streamlit will print the local URL (typically `http://localhost:8501`) and automatically open your default web browser.

### Features in the Web App:
- **🏠 Home & Overview**: Context of diabetes, Pima database origins, target definition, pipeline structure, and medical safety notices.
- **📊 Data Exploration**: Raw vs. cleaned data tables, invalid zero charts, histograms/KDEs split by diabetes outcome, and interactive correlation heatmaps.
- **📈 Model Evaluation**: Cross-validation vs. test set comparison tables, interactive Confusion Matrices, ROC curves with AUC indicators, and Precision-Recall curves.
- **🔮 Diabetes Risk Prediction**: Input sliders and number fields with reference ranges, model selector, probability gauges, key risk factor highlights, and clinical disclaimers.

---

## 🛠️ Troubleshooting Common Issues

### 1. `python` or `pip` points to an older version (e.g. Python 2.7)
- **Symptom**: `SyntaxError` or module not found errors when running `python`.
- **Fix**: Use the specific Python 3 path or `py -3`. If Anaconda is installed on Windows:
  ```powershell
  & "C:\ProgramData\anaconda3\python.exe" -m streamlit run app.py
  ```

### 2. `Port 8501 is already in use`
- **Symptom**: Streamlit warns that port 8501 is occupied.
- **Fix**: Launch Streamlit on a custom port:
  ```bash
  python -m streamlit run app.py --server.port 8502
  ```

### 3. Missing `data/diabetes.csv` or `models/all_models.joblib`
- **Symptom**: `FileNotFoundError: Dataset not found` or `Model artifacts not found`.
- **Fix**: Run `python train_model.py` from the project root directory.

---

## 🛡️ Safety, Ethical & Clinical Disclaimers

> [!WARNING]
> ### Educational Demonstration Only
> This application is strictly an educational tool demonstrating applied machine learning workflows.
> - **Not a Diagnostic Tool**: It is **NOT** a certified medical device, medical diagnostic instrument, or clinical decision support software.
> - **Do Not Base Treatment On Outputs**: Under no circumstances should individual patients or clinicians use predictions generated by this software for medical diagnosis, drug dosing, or clinical decision-making.
> - **Demographic Bias**: The underlying dataset represents adult females of Pima Indian heritage. Clinical patterns, baseline glucose distributions, and body mass indicators differ significantly across different racial, ethnic, age, and sex cohorts.
> - **Need for External Validation**: Machine learning models in medicine require thorough multi-center prospective clinical trials and regulatory authorization before any real-world clinical application.
