
import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.pkl")

st.title("Allstate Claims Severity Predictor 💡")
st.write("Enter continuous feature values (cont1–cont14) to estimate insurance loss.")

inputs = []
for i in range(1, 15):
    val = st.number_input(f"cont{i}", min_value=0.0, max_value=1.0, step=0.01)
    inputs.append(val)

if st.button("Predict Loss"):
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict(X)[0]
    st.success(f"Predicted Loss: **${pred:,.2f}**")
