import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide"
)

NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

FEATURES = NUMERIC + CATEGORICAL


@st.cache_resource
def train_model():
    df = pd.read_csv("credit_risk_dataset.csv")

    missing = set(FEATURES + ["loan_status"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.drop_duplicates()
    df = df[
        (df["person_age"] >= 18)
        & (df["person_age"] <= 100)
        & (df["person_income"] > 0)
        & (df["loan_amnt"] > 0)
        & (df["person_emp_length"].isna()
           | (df["person_emp_length"] <= df["person_age"]))
    ].copy()

    X = df[FEATURES]
    y = df["loan_status"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant", fill_value="Missing"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1),
            eval_metric="logloss",
            random_state=42,
            n_jobs=2
        ))
    ])

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    f1 = f1_score(y_test, (probabilities >= 0.5).astype(int))

    return model, auc, f1


st.title("🏦 Explainable Credit Risk Assessment")
st.caption(
    "Machine learning prototype for loan-default risk estimation"
)

st.warning(
    "Educational prototype only. Predictions are estimates, not "
    "financial advice or a decision to approve or reject a loan."
)

try:
    with st.spinner("Preparing the credit-risk model on first launch..."):
        model, auc, f1 = train_model()
except Exception as exc:
    st.error(f"Unable to load the dataset or train the model: {exc}")
    st.stop()

m1, m2 = st.columns(2)
m1.metric("Test ROC-AUC", f"{auc:.3f}")
m2.metric("Test F1 score", f"{f1:.3f}")

st.subheader("Applicant details")

with st.form("risk_form"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age (years)", 18, 100, 28)
        income = st.number_input(
            "Annual income", min_value=1, value=600000, step=10000
        )
        employment = st.number_input(
            "Employment length (years)", 0.0, 80.0, 4.0, 0.5
        )
        home = st.selectbox(
            "Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"]
        )
        intent = st.selectbox(
            "Loan purpose",
            [
                "PERSONAL", "EDUCATION", "MEDICAL",
                "VENTURE", "HOMEIMPROVEMENT",
                "DEBTCONSOLIDATION"
            ]
        )

    with c2:
        amount = st.number_input(
            "Loan amount", min_value=1, value=100000, step=5000
        )
        rate = st.number_input(
            "Interest rate (%)", 5.0, 30.0, 12.0, 0.25
        )
        grade = st.selectbox(
            "Loan grade", list("ABCDEFG")
        )
        history = st.number_input(
            "Credit history length (years)", 0, 60, 5
        )
        previous_default = st.selectbox(
            "Historical default on file", ["N", "Y"]
        )

    submitted = st.form_submit_button(
        "Assess credit risk", type="primary", use_container_width=True
    )

if submitted:
    if employment > age or history > age:
        st.error("Please check employment and credit history lengths.")
    else:
        applicant = pd.DataFrame([{
            "person_age": age,
            "person_income": income,
            "person_emp_length": employment,
            "loan_amnt": amount,
            "loan_int_rate": rate,
            "loan_percent_income": amount / income,
            "cb_person_cred_hist_length": history,
            "person_home_ownership": home,
            "loan_intent": intent,
            "loan_grade": grade,
            "cb_person_default_on_file": previous_default
        }])[FEATURES]

        probability = float(model.predict_proba(applicant)[0, 1])

        st.subheader("Prediction")
        st.metric("Estimated default probability", f"{probability:.1%}")
        st.progress(probability)

        if probability >= 0.5:
            st.error("Higher estimated default risk (model threshold: 50%).")
        else:
            st.success("Lower estimated default risk (model threshold: 50%).")

        st.caption(
            "The 50% cutoff is a demonstration threshold, not a validated "
            "lending policy. A lower or higher cutoff changes the results."
        )

        st.subheader("Global feature importance")
        prep = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        names = prep.get_feature_names_out()

        importance = pd.Series(
            classifier.feature_importances_, index=names
        ).sort_values(ascending=False).head(10).sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        importance.plot(kind="barh", ax=ax)
        ax.set_xlabel("XGBoost feature importance")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
