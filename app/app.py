
import streamlit as st, joblib, numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart — Sales Forecast")
st.markdown("Predict **Product Store Sales Total** using the trained Gradient Boosting model.")

@st.cache_resource
def load_artifacts():
    model    = joblib.load(hf_hub_download("karamvir-singh/superkart-gbm", "gbm_model.pkl"))
    encoders = joblib.load(hf_hub_download("karamvir-singh/superkart-gbm", "label_encoders.pkl"))
    return model, encoders

model, encoders = load_artifacts()

with st.form("predict"):
    c1, c2 = st.columns(2)
    with c1:
        weight = st.number_input("Weight (kg)", 4.0, 22.0, 12.6, 0.1)
        mrp    = st.number_input("MRP", 31.0, 266.0, 147.0, 0.5)
        area   = st.slider("Allocated Area Ratio", 0.004, 0.298, 0.06, 0.001)
    with c2:
        sugar = st.selectbox("Sugar Content", ["Low Sugar", "No Sugar", "Regular"])
        ptype = st.selectbox("Product Type", list(encoders["Product_Type"].classes_))
    c3, c4 = st.columns(2)
    with c3:
        ssize = st.selectbox("Store Size", ["High", "Medium", "Small"])
        stype = st.selectbox("Store Type", list(encoders["Store_Type"].classes_))
    with c4:
        city = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
        age  = st.slider("Store Age (years)", 15, 37, 20)
    if st.form_submit_button("Predict"):
        row = {
            "Product_Weight": weight,
            "Product_Sugar_Content": encoders["Product_Sugar_Content"].transform([sugar])[0],
            "Product_Allocated_Area": area,
            "Product_Type": encoders["Product_Type"].transform([ptype])[0],
            "Product_MRP": mrp,
            "Store_Size": encoders["Store_Size"].transform([ssize])[0],
            "Store_Location_City_Type": encoders["Store_Location_City_Type"].transform([city])[0],
            "Store_Type": encoders["Store_Type"].transform([stype])[0],
            "Store_Age": age,
        }
        pred = model.predict(pd.DataFrame([row]))[0]
        st.success(f"Predicted Sales: Rs {pred:,.2f}")
