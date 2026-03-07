import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.cluster.hierarchy import fcluster

st.set_page_config(page_title="Pet Adoption Models", layout="wide")

st.title("Pet Adoption Prediction UI")
st.write("Use the controls on the left to select a record and see predictions from each trained model.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("petfinder-mini.csv")
    df = df.drop(columns=["Name", "PetID", "Description"], errors="ignore")
    df.fillna(0, inplace=True)
    return df

@st.cache_data
def load_models():
    models = {}
    try:
        models["rf"] = joblib.load("rf_model.joblib")
    except FileNotFoundError:
        models["rf"] = None
    try:
        models["logistic"] = joblib.load("logistic_model.joblib")
    except FileNotFoundError:
        models["logistic"] = None
    try:
        models["kmeans"] = joblib.load("kmeans_model.joblib")
    except FileNotFoundError:
        models["kmeans"] = None
    try:
        hierarchical = joblib.load("hierarchical_model.joblib")
        models["hierarchical"] = hierarchical
    except FileNotFoundError:
        models["hierarchical"] = None
    return models

@st.cache_data
def preprocess_df(df):
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df

# Load everything
df = load_data()
models = load_models()
preproc_df = preprocess_df(df)

# Map numerical labels to readable class names
adoption_speed_map = {
    0: "Adopted same day",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 2 months",
    4: "Adopted after 2+ months",
}
adopted_map = {0: "Not Adopted", 1: "Adopted"}

cluster_name = lambda c: f"Cluster {chr(65 + (c % 26))}"  # A, B, C... for cluster numbers

def format_adoption_speed(pred):
    return adoption_speed_map.get(pred, f"Class {pred}")

def format_adopted(pred):
    return adopted_map.get(pred, f"Class {pred}")

if models["rf"] is None or models["logistic"] is None or models["kmeans"] is None or models["hierarchical"] is None:
    st.warning("One or more model files are missing. Make sure you ran the training notebooks and generated: rf_model.joblib, logistic_model.joblib, kmeans_model.joblib, hierarchical_model.joblib.")

mode = st.sidebar.selectbox("Input mode", ["Use dataset row", "Custom input"])

features = preproc_df.drop(columns=["AdoptionSpeed"], errors="ignore").columns.tolist()
feature_defaults = preproc_df[features].mean().to_dict()

col1, col2 = st.columns([1, 3])

with col1:
    if mode == "Use dataset row":
        st.subheader("Select sample")
        idx = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
        st.write("Total rows:", len(df))
        show_raw = st.checkbox("Show raw row data", value=True)

    else:
        st.subheader("Custom input")
        custom_values = {}
        for feat in features:
            custom_values[feat] = st.number_input(feat, value=float(feature_defaults[feat]))

with col2:
    st.subheader("Model Predictions")

    if mode == "Use dataset row":
        sample = preproc_df.drop(columns=["AdoptionSpeed"], errors="ignore").iloc[[idx]]
        if show_raw:
            row = df.iloc[[idx]]
            st.write("Selected row")
            st.dataframe(row)
    else:
        sample = pd.DataFrame([custom_values])
        st.write("Custom feature input")
        st.dataframe(sample)

    if models["rf"] is not None:
        pred = int(models["rf"].predict(sample)[0])
        st.write("**Random Forest (AdoptionSpeed):**", format_adoption_speed(pred))
    else:
        st.write("**Random Forest:** model file missing")

    if models["logistic"] is not None:
        pred = int(models["logistic"].predict(sample)[0])
        st.write("**Logistic (Adopted binary):**", format_adopted(pred))
    else:
        st.write("**Logistic Regression:** model file missing")

    if models["kmeans"] is not None:
        pred = int(models["kmeans"].predict(sample)[0])
        st.write("**KMeans cluster:**", cluster_name(pred))
    else:
        st.write("**KMeans:** model file missing")

    if models["hierarchical"] is not None:
        linked = models["hierarchical"]["linked"]
        subset = models["hierarchical"]["subset"]
        if mode == "Use dataset row" and idx < subset.shape[0]:
            labels = fcluster(linked, t=4, criterion="maxclust")
            pred = int(labels[idx])
            st.write("**Hierarchical cluster (first {} rows):**".format(subset.shape[0]), cluster_name(pred))
        else:
            st.write("**Hierarchical cluster:** out of range for the stored subset")
    else:
        st.write("**Hierarchical:** model file missing")

st.markdown("---")
st.write("If the models are missing, run the training notebooks to generate the `.joblib` files in this folder.")
