import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model and tools
model = tf.keras.models.load_model("churn_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
ct = pickle.load(open("column_transformer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.title("Customer Churn Prediction")

credit = st.number_input("Credit Score", 300, 900)
geo = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure", 0, 10)
balance = st.number_input("Balance")
products = st.number_input("Products", 1, 4)
card = st.selectbox("Has Credit Card", [0,1])
active = st.selectbox("Is Active Member", [0,1])
salary = st.number_input("Salary")

if st.button("Predict"):
    data = np.array([[credit, geo, gender, age, tenure,
                      balance, products, card, active, salary]])

    data[:,2] = le.transform(data[:,2])
    data = ct.transform(data)
    data = scaler.transform(data)

    pred = model.predict(data)

    if pred[0][0] > 0.5:
        st.error("Customer will leave ❌")
    else:
        st.success("Customer will stay ✅")