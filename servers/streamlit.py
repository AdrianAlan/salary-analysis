import streamlit as st
import requests as rq
import json

st.title("Predict Expected Yearly Total Compensation")
location = st.selectbox("Location", options = ['United States', 'Not United States'])
company = st.text_input("Company")
yoe = st.slider("Years of Experience", 0, 50, 1)
yac = st.slider("Years at Company", 0, 50, 1)
sex = st.selectbox("Sex", options=['Male' , 'Female'])
prediction_button = st.button("Predict")

if prediction_button:
    response = rq.post(
            'http://adrianalan.pythonanywhere.com/predict',
            json={
            "location": location,
            "company": company,
            "yoe": yoe,
            "yac": yac,
            "sex":sex
            }
        )
    prediction = json.loads(response.content)
    st.write("Your expected target salary is ${:,.0f}.".format(prediction['expected']*1000))
    

