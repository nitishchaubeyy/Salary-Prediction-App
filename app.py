
import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction App")
st.divider()

st.write("With this app, you can get estimations for the salaries of the company's employees.")


try:
    model = joblib.load("linearmodel.pkl") 
except FileNotFoundError:
    st.error("Model file 'linearmodel.pkl' not found! Please make sure you have created and saved the model in a previous step.")
    st.stop()

years = st.number_input("Enter the years at the company", value=1, step=1, min_value=0)
job_rate = st.number_input("Enter the job rate", value=3.5, step=0.5, min_value=0.0)

predict_button = st.button("Press for salary prediction")
st.divider()

if predict_button:
    st.balloons()
    X = np.array([[years, job_rate]])
    prediction = model.predict(X)
    st.write("Salary prediction is:")
    st.write(f"{prediction[0]:.2f}")
else:
    st.write("Please press the button for the app to make the prediction.")
