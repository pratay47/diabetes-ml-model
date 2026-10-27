"""
app.py
======
Type 2 Diabetes Prediction System - Interactive Streamlit Web Application

Provides:
- Home: Project overview, clinical dataset source, target definition, and ethical disclaimers.
- Data Exploration: Exploratory data analysis, missing/invalid zero analysis, and distributions.
- Model Evaluation: Comparison of Logistic Regression, Decision Tree, and Random Forest.
- Risk Prediction: Interactive risk prediction with input validation, probability, and limitations.
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Configure Streamlit page settings
st.set_page_config(
    page_title="Type 2 Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, beginner-friendly UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-emerald { background-color: #D1FAE5; color: #065F46; }
    .card-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #991B1B;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #065F46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
DATA_PATH = os.path.join("data", "diabetes.csv")
MODELS_DIR = "models"
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")


@st.cache_data
def load_data():
    """Load raw dataset and create cleaned version with zeros replaced by NaN for clinical variables."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Please run `python train_model.py` first.")
        st.stop()
    df_raw = pd.read_csv(DATA_PATH)

    invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df_clean = df_raw.copy()
    for col in invalid_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    return df_raw, df_clean


@st.cache_resource
def load_model_artifacts():
    """Load trained models and evaluation artifacts."""
    if not os.path.exists(ALL_MODELS_PATH):
        st.error(f"Model artifacts not found at {ALL_MODELS_PATH}. Please run `python train_model.py` first.")
        st.stop()
    artifacts = joblib.load(ALL_MODELS_PATH)
    return artifacts


# Load resources
df_raw, df_clean = load_data()
artifacts = load_model_artifacts()
models = artifacts["models"]
best_model_name = artifacts["best_model_name"]
feature_stats = artifacts["feature_stats"]
feature_names = artifacts["feature_names"]
cv_results = artifacts["cv_results"]
test_results = artifacts["test_results"]
zero_counts = artifacts["zero_counts"]

# Sidebar Navigation
st.sidebar.markdown("## 🩺 Navigation")
page = st.sidebar.radio(
    "Go to page:",
    [
        "🏠 Home & Overview",
        "📊 Data Exploration",
        "📈 Model Evaluation",
        "🔮 Diabetes Risk Prediction",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Pipeline Info")
st.sidebar.markdown(f"**Baseline Model:** Logistic Regression")
st.sidebar.markdown(f"**Selected Best (CV):** `{best_model_name}`")
st.sidebar.markdown(f"**Dataset Size:** 768 samples (8 features)")
st.sidebar.markdown(
    """
    <div style="font-size:0.8rem; color:#64748B; margin-top:1rem;">
    ⚠️ <b>Disclaimer:</b> Educational tool only. Not certified for clinical diagnostics.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: HOME & OVERVIEW
# ==============================================================================
if page == "🏠 Home & Overview":
    st.markdown('<div class="main-title">Type 2 Diabetes Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">A Beginner-Friendly Machine Learning Project with Scikit-Learn & Streamlit</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="badge-pill badge-blue">Python 3</span>
        <span class="badge-pill badge-blue">Scikit-Learn</span>
        <span class="badge-pill badge-emerald">Stratified Cross-Validation</span>
        <span class="badge-pill badge-amber">Leakage-Free Pipeline</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Safety & Ethical Callout
    st.warning(
        """
        **⚠️ Important Clinical & Educational Notice**:
        This application is an educational data science project built for instructional purposes.
        Predictions generated here are **probabilistic approximations from a historical dataset** and
        **MUST NOT** be interpreted as clinical diagnoses or medical advice. Consult qualified healthcare
        professionals for medical screenings and diagnostic decisions.
        """
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🎯 Project Purpose")
        st.write(
            """
            Type 2 diabetes mellitus is a chronic metabolic disorder characterized by high blood glucose levels,
            insulin resistance, and relative lack of insulin. Early identification of individuals at high risk allows
            for timely lifestyle interventions and medical monitoring.

            This project demonstrates a rigorous, leak-free machine learning workflow for binary classification:
            - **Data Auditing**: Identifying biologically impossible zero values (e.g. Glucose = 0 or BMI = 0) representing hidden missing values.
            - **Leak-Free Preprocessing**: Encapsulating median imputation and feature scaling inside Scikit-Learn Pipelines.
            - **Model Comparison**: Benchmarking a **Logistic Regression** baseline against **Decision Tree** and **Random Forest** classifiers.
            - **Principled Evaluation**: Selecting the optimal model using **5-Fold Stratified Cross-Validation** on training data, avoiding test set data snooping.
            """
        )

        st.markdown("### 🔬 Dataset Origin & Target Definition")
        st.write(
            """
            - **Dataset Source**: *Pima Indians Diabetes Database* (PIDD), originally gathered by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), hosted on the UCI Machine Learning Repository and Kaggle.
            - **Study Reference**: Smith, C.W. et al. (1988), *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*.
            - **Target Variable (`Outcome`)**:
              - `0` (Negative): Tested negative for diabetes onset.
              - `1` (Positive): Tested positive for diabetes onset (within 5 years according to WHO diagnostic criteria).
            - **Target Distribution**: 500 Negative (65.1%) and 268 Positive (34.9%), presenting moderate class imbalance.
            """
        )

    with col2:
        st.markdown("### ⚠️ Key Dataset Limitations & Bias")
        st.info(
            """
            1. **Demographic Specificity**:
               All 768 participants are adult females (aged $\\ge 21$) of Pima Native American heritage living near Phoenix, Arizona.
               *It cannot be generalized to men, other ethnic backgrounds, or pediatric populations without external validation.*

            2. **Disguised Missing Values**:
               Medical measurements of 0 for Insulin (48.7%), Skin Thickness (29.6%), Blood Pressure (4.6%), BMI (1.4%), and Glucose (0.7%) are physiological impossibilities representing unrecorded tests.

            3. **Historical Context**:
               Collected several decades ago; modern primary diagnostic biomarkers (such as HbA1c) are absent from this feature set.

            4. **Need for External Validation**:
               Before any ML model can be considered in healthcare, it must undergo multi-center external validation and algorithmic bias auditing.
            """
        )

    st.markdown("### 📐 Feature Summary")
    feature_desc = pd.DataFrame([
        {"Feature": "Pregnancies", "Type": "Integer", "Clinical Meaning": "Number of times pregnant (0 is physiologically valid)"},
        {"Feature": "Glucose", "Type": "Continuous (mg/dL)", "Clinical Meaning": "Plasma glucose concentration at 2 hours in an oral glucose tolerance test"},
        {"Feature": "BloodPressure", "Type": "Continuous (mm Hg)", "Clinical Meaning": "Diastolic blood pressure"},
        {"Feature": "SkinThickness", "Type": "Continuous (mm)", "Clinical Meaning": "Triceps skin fold thickness (proxy for subcutaneous fat)"},
        {"Feature": "Insulin", "Type": "Continuous (μU/mL)", "Clinical Meaning": "2-Hour serum insulin"},
        {"Feature": "BMI", "Type": "Continuous (kg/m²)", "Clinical Meaning": "Body Mass Index (Weight in kg / Height in m²)"},
        {"Feature": "DiabetesPedigreeFunction", "Type": "Continuous Score", "Clinical Meaning": "Genetic risk score based on family diabetes history"},
        {"Feature": "Age", "Type": "Integer (Years)", "Clinical Meaning": "Age of participant (minimum 21 years)"},
    ])
    st.dataframe(feature_desc, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: DATA EXPLORATION (EDA)
# ==============================================================================
elif page == "📊 Data Exploration":
    st.markdown('<div class="main-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspecting Distributions, Hidden Missingness, and Correlations</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Data Overview & Missing Values", "📊 Feature Distributions", "🔗 Correlation & Relationships"])

    with tab1:
        st.markdown("#### 1. Dataset Overview")
        st.write(f"The dataset contains **{df_raw.shape[0]} rows** and **{df_raw.shape[1]} columns**.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**First 5 rows of raw data:**")
            st.dataframe(df_raw.head(), use_container_width=True)
        with col_b:
            st.markdown("**Summary Statistics (Cleaned Data with NaNs for Invalid Zeros):**")
            st.dataframe(df_clean.describe().round(2), use_container_width=True)

        st.markdown("#### 2. The 'Invalid Zeros' Problem")
        st.write(
            """
            In medical datasets, unrecorded measurements are sometimes recorded as `0` instead of `null`.
            For example, a living person cannot have a plasma glucose of 0 or a diastolic blood pressure of 0!
            Below is the audit of disguised missing values:
            """
        )

        missing_data = []
        for col, count in zero_counts.items():
            pct = count / len(df_raw) * 100
            missing_data.append({"Feature": col, "Invalid Zero Count": count, "Missing Percentage (%)": f"{pct:.1f}%"})
        st.table(pd.DataFrame(missing_data))

        # Chart of Missing Values
        fig, ax = plt.subplots(figsize=(8, 3))
        cols = list(zero_counts.keys())
        counts = [zero_counts[c] for c in cols]
        sns.barplot(x=cols, y=counts, palette="Blues_r", ax=ax)
        ax.set_ylabel("Number of Invalid 0s")
        ax.set_title("Biologically Impossible Zero Values in Features")
        for i, v in enumerate(counts):
            ax.text(i, v + 5, f"{v} ({v/len(df_raw)*100:.1f}%)", ha="center", fontsize=9)
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.markdown("#### Feature Distributions by Diabetes Outcome")
        st.write("Compare the distributions of clinical features between non-diabetic (`Outcome=0`) and diabetic (`Outcome=1`) patients.")

        selected_feature = st.selectbox("Select feature to visualize:", feature_names, index=1)

        col_c, col_d = st.columns(2)

        with col_c:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(
                data=df_clean,
                x=selected_feature,
                hue="Outcome",
                kde=True,
                palette={0: "#3B82F6", 1: "#EF4444"},
                ax=ax,
                bins=25,
            )
            ax.set_title(f"Histogram & KDE of {selected_feature}")
            ax.legend(["Diabetic (1)", "Non-Diabetic (0)"])
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

        with col_d:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df_clean,
                x="Outcome",
                y=selected_feature,
                palette={0: "#93C5FD", 1: "#FCA5A5"},
                ax=ax,
            )
            ax.set_xticklabels(["Non-Diabetic (0)", "Diabetic (1)"])
            ax.set_title(f"Boxplot of {selected_feature} by Outcome")
            sns.despine()
            st.pyplot(fig)
            plt.close(fig)

        # Brief clinical note on feature
        if selected_feature == "Glucose":
            st.info("💡 **Clinical Observation**: Notice the marked shift to higher glucose values in diabetic patients. Glucose is typically the strongest single linear predictor of diabetes.")
        elif selected_feature == "BMI":
            st.info("💡 **Clinical Observation**: Higher BMI shows a clear upward shift in the diabetic cohort, aligning with established links between obesity and insulin resistance.")
        elif selected_feature == "Age":
            st.info("💡 **Clinical Observation**: The non-diabetic cohort is heavily skewed toward younger participants (20-30), whereas risk increases noticeably beyond age 30.")

    with tab3:
        st.markdown("#### Feature Correlation Heatmap")
        st.write("Pearson correlation matrix between features and target variable `Outcome`.")

        corr = df_clean.corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, vmin=-1, vmax=1)
        ax.set_title("Correlation Matrix (Cleaned Data with Validated Biological Values)")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Outcome Correlation Ranking")
        outcome_corr = corr["Outcome"].drop("Outcome").sort_values(ascending=False)
        st.dataframe(pd.DataFrame({"Pearson Correlation with Outcome": outcome_corr.round(3)}))


# ==============================================================================
# PAGE 3: MODEL EVALUATION & COMPARISON
# ==============================================================================
elif page == "📈 Model Evaluation":
    st.markdown('<div class="main-title">Model Evaluation & Benchmark Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comparing Logistic Regression (Baseline), Decision Tree, and Random Forest</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🧠 Methodological Note: Avoiding Test Set Data Snooping
        In machine learning, it is poor practice to repeatedly test multiple models or hyperparameters on the **test set**
        and pick the one that scores highest on that test set. Doing so effectively "fits" the choice to the test set,
        causing optimistic bias and compromising generalization measurement.

        **Our Approach:**
        1. We evaluate all models with **5-Fold Stratified Cross-Validation strictly on the Training Set (80%)**.
        2. We select the winning model based on **Training Cross-Validation F1 and ROC-AUC scores**.
        3. We then evaluate once on the **Held-Out Test Set (20%)** as an unbiased assessment of unseen performance.
        """
    )

    st.markdown("---")

    # Tables of Metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Cross-Validation Results (Training Set)")
        st.caption("Mean score ± standard deviation across 5 stratified folds")
        cv_table_data = []
        for m_name, metrics in cv_results.items():
            cv_table_data.append({
                "Model": m_name,
                "Accuracy": f"{metrics['accuracy']['mean']:.3f} ± {metrics['accuracy']['std']:.2f}",
                "Precision": f"{metrics['precision']['mean']:.3f} ± {metrics['precision']['std']:.2f}",
                "Recall": f"{metrics['recall']['mean']:.3f} ± {metrics['recall']['std']:.2f}",
                "F1 Score": f"{metrics['f1']['mean']:.3f} ± {metrics['f1']['std']:.2f}",
                "ROC-AUC": f"{metrics['roc_auc']['mean']:.3f} ± {metrics['roc_auc']['std']:.2f}",
            })
        st.dataframe(pd.DataFrame(cv_table_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### 2. Held-Out Test Set Performance (20% Test)")
        st.caption("Evaluated on 154 unseen patient samples (100 Negative, 54 Positive)")
        test_table_data = []
        for m_name, metrics in test_results.items():
            test_table_data.append({
                "Model": m_name,
                "Accuracy": f"{metrics['accuracy']:.3f}",
                "Precision": f"{metrics['precision']:.3f}",
                "Recall": f"{metrics['recall']:.3f}",
                "F1 Score": f"{metrics['f1']:.3f}",
                "ROC-AUC": f"{metrics['roc_auc']:.3f}",
            })
        st.dataframe(pd.DataFrame(test_table_data), hide_index=True, use_container_width=True)

    # Explanation of trade-offs
    st.info(
        f"""
        **🏆 Selected Optimal Model: `{best_model_name}`**
        - **Why Logistic Regression won cross-validation**: On small, moderately noisy tabular biomedical datasets (768 rows), regularized linear models often generalize better and with less variance than complex tree models.
        - **Why Recall is Critical in Healthcare**: A **False Negative** (predicting non-diabetic when the patient actually has diabetes) means a patient misses early intervention. A **False Positive** merely triggers a follow-up lab test. Therefore, optimizing for **Recall and F1 Score** is clinically preferable over optimizing raw accuracy alone.
        """
    )

    st.markdown("---")
    st.markdown("### 📊 Visual Diagnostic Curves")

    tab_roc, tab_cm, tab_pr = st.tabs(["📈 ROC Curves", "🎯 Confusion Matrices", "🔍 Precision-Recall Curves"])

    with tab_roc:
        st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
        st.write("Plots the True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) at various classification thresholds.")

        fig, ax = plt.subplots(figsize=(7, 5))
        for m_name, res in test_results.items():
            fpr = res["roc_curve"]["fpr"]
            tpr = res["roc_curve"]["tpr"]
            auc_val = res["roc_auc"]
            ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", lw=2)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
        ax.set_title("Test Set ROC Curves")
        ax.legend(loc="lower right")
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)

    with tab_cm:
        st.markdown("#### Test Set Confusion Matrices")
        cols_cm = st.columns(3)
        for i, (m_name, res) in enumerate(test_results.items()):
            with cols_cm[i]:
                st.markdown(f"**{m_name}**")
                cm = np.array(res["confusion_matrix"])
                fig, ax = plt.subplots(figsize=(4, 3.5))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["Actual 0", "Actual 1"],
                    ax=ax,
                )
                ax.set_title(f"{m_name}")
                st.pyplot(fig)
                plt.close(fig)
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    with tab_pr:
        st.markdown("#### Precision-Recall Curves")
        st.write("Crucial metric when evaluating imbalanced datasets, emphasizing the minority positive class.")
        fig, ax = plt.subplots(figsize=(7, 5))
        for m_name, res in test_results.items():
            p = res["pr_curve"]["precision"]
            r = res["pr_curve"]["recall"]
            ax.plot(r, p, label=f"{m_name}", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Test Set Precision-Recall Curves")
        ax.legend(loc="lower left")
        sns.despine()
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# PAGE 4: PREDICTION PAGE
# ==============================================================================
elif page == "🔮 Diabetes Risk Prediction":
    st.markdown('<div class="main-title">Interactive Diabetes Risk Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter patient clinical metrics to estimate statistical risk of Type 2 Diabetes</div>', unsafe_allow_html=True)

    # Prominent Disclaimer
    st.markdown(
        """
        <div class="card-box" style="border-left: 5px solid #F59E0B;">
        <b>⚠️ Educational Demonstration Only:</b> This risk screener uses historical statistical models trained on
        adult females of Pima Indian heritage. It is not an authorized diagnostic medical device and does not substitute
        for blood tests, clinical examinations, or professional medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model selector
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        chosen_model_name = st.selectbox(
            "Select inference model:",
            options=list(models.keys()),
            index=list(models.keys()).index(best_model_name),
            help="Choose between Logistic Regression (Baseline / Best CV), Decision Tree, or Random Forest.",
        )
    with col_sel2:
        st.markdown(f"**Selected Model:** `{chosen_model_name}`")
        if chosen_model_name == best_model_name:
            st.success("⭐ Recommended: Highest CV F1/AUC")

    selected_pipeline = models[chosen_model_name]

    st.markdown("### 📋 Patient Clinical Measurements")
    st.caption("Adjust sliders or enter numbers based on clinical lab readings. Default values reflect dataset medians.")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        pregnancies = st.slider(
            "Number of Pregnancies",
            min_value=0,
            max_value=17,
            value=int(feature_stats["Pregnancies"]["median"]),
            step=1,
            help="Number of times pregnant (0 is normal for non-pregnant/nulliparous individuals).",
        )

        glucose = st.number_input(
            "2-Hour Oral Glucose Tolerance (mg/dL)",
            min_value=40.0,
            max_value=300.0,
            value=float(feature_stats["Glucose"]["median"]),
            step=1.0,
            help="Plasma glucose concentration 2 hours after a 75g oral glucose load. Normal: <140 mg/dL; Prediabetes: 140-199 mg/dL; Diabetes: ≥200 mg/dL.",
        )

        blood_pressure = st.number_input(
            "Diastolic Blood Pressure (mm Hg)",
            min_value=30.0,
            max_value=150.0,
            value=float(feature_stats["BloodPressure"]["median"]),
            step=1.0,
            help="Diastolic blood pressure reading. Normal is generally <80 mm Hg.",
        )

        skin_thickness = st.number_input(
            "Triceps Skinfold Thickness (mm)",
            min_value=5.0,
            max_value=100.0,
            value=float(feature_stats["SkinThickness"]["median"]),
            step=1.0,
            help="Thickness of triceps skin fold, representing subcutaneous body fat.",
        )

    with form_col2:
        insulin = st.number_input(
            "2-Hour Serum Insulin (μU/mL)",
            min_value=10.0,
            max_value=900.0,
            value=float(feature_stats["Insulin"]["median"]),
            step=1.0,
            help="Serum insulin level 2 hours after glucose challenge.",
        )

        bmi = st.number_input(
            "Body Mass Index (BMI in kg/m²)",
            min_value=15.0,
            max_value=70.0,
            value=float(round(feature_stats["BMI"]["median"], 1)),
            step=0.1,
            help="Weight in kg divided by height in meters squared. Normal: 18.5-24.9; Overweight: 25-29.9; Obese: ≥30.",
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function (Genetic Score)",
            min_value=0.05,
            max_value=2.50,
            value=float(round(feature_stats["DiabetesPedigreeFunction"]["median"], 3)),
            step=0.01,
            help="A synthesis of diabetes history in first- and second-degree relatives.",
        )

        age = st.slider(
            "Age (Years)",
            min_value=21,
            max_value=90,
            value=int(feature_stats["Age"]["median"]),
            step=1,
            help="Age of individual. Note that the original study enrolled individuals aged 21 and above.",
        )

    # Biological plausibility validation check
    validation_warnings = []
    if glucose < 50:
        validation_warnings.append("Glucose reading is very low (hypoglycemia range).")
    elif glucose > 220:
        validation_warnings.append("Glucose reading is severely elevated.")
    if blood_pressure < 50:
        validation_warnings.append("Blood pressure reading is unusually low for diastolic BP.")
    if bmi < 16.0 or bmi > 55.0:
        validation_warnings.append("BMI reading is in an extreme clinical range.")

    for w in validation_warnings:
        st.warning(f"⚠️ Range Notice: {w}")

    # Predict Button
    st.markdown("---")
    predict_clicked = st.button("🔍 Generate Risk Assessment", type="primary", use_container_width=True)

    if predict_clicked:
        # Prepare input dataframe matching exact feature names
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }])

        # Perform inference using the pipeline (handles imputation and scaling internally)
        prediction = selected_pipeline.predict(input_data)[0]
        probabilities = selected_pipeline.predict_proba(input_data)[0]
        risk_score = probabilities[1] * 100.0  # Class 1 probability as percentage

        st.markdown("### 📊 Assessment Results")

        res_col1, res_col2 = st.columns([3, 2])

        with res_col1:
            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="risk-high">
                        <h3 style="margin-top:0;">⚠️ Elevated Statistical Risk Detected</h3>
                        <p style="font-size:1.1rem; margin-bottom:0.5rem;">
                            Estimated Probability of Diabetes Onset: <b>{risk_score:.1f}%</b>
                        </p>
                        <p style="margin-bottom:0;">
                            Based on the entered indicators and the <b>{chosen_model_name}</b> model,
                            the patient's clinical profile aligns with characteristics of the diabetic cohort.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-low">
                        <h3 style="margin-top:0;">✅ Low Statistical Risk Indicated</h3>
                        <p style="font-size:1.1rem; margin-bottom:0.5rem;">
                            Estimated Probability of Diabetes Onset: <b>{risk_score:.1f}%</b>
                        </p>
                        <p style="margin-bottom:0;">
                            Based on the entered indicators and the <b>{chosen_model_name}</b> model,
                            the profile aligns more closely with the non-diabetic cohort.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown(f"**Model Confidence Breakdown:**")
            st.progress(float(risk_score / 100.0))
            p_col_a, p_col_b = st.columns(2)
            p_col_a.metric("Non-Diabetic Probability (Class 0)", f"{probabilities[0]*100:.1f}%")
            p_col_b.metric("Diabetic Risk Probability (Class 1)", f"{probabilities[1]*100:.1f}%")

        with res_col2:
            st.markdown("#### 🔬 Key Factors in this Profile")
            st.write(
                f"""
                - **Glucose ({glucose} mg/dL):** {'Elevated (>140 mg/dL)' if glucose >= 140 else 'Normal range'}
                - **BMI ({bmi} kg/m²):** {'Obese category (≥30)' if bmi >= 30 else ('Overweight (25-30)' if bmi >= 25 else 'Normal (<25)')}
                - **Age ({age} yrs):** {'Increased age risk factor (>35)' if age > 35 else 'Younger cohort'}
                - **Genetic Score ({pedigree}):** {'Higher hereditary risk factor' if pedigree > 0.6 else 'Moderate/Average hereditary factor'}
                """
            )

        st.markdown("---")
        st.markdown("#### 🛡️ Critical Clinical Guidance & Model Limitations")
        st.write(
            """
            - **No Individual Calibration**: In machine learning, output probabilities are mathematical confidence scores of the classifier, not true calibrated absolute epidemiological risk for an arbitrary patient.
            - **Demographic Boundary**: This model was trained on female individuals of Pima Indian ancestry. Physiological markers vary across populations, and results cannot be assumed valid for other groups.
            - **Next Steps**: Any clinical concern should prompt standard laboratory diagnostics, including fasting plasma glucose, oral glucose tolerance testing (OGTT), or glycated hemoglobin (HbA1c) measured in an accredited medical facility.
            """
        )
