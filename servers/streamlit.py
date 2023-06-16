import streamlit as st
import requests as rq
import json
import streamlit as st
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection


st.title("Estimate Your Total Yearly Compensation")
location = st.text_input("Location")
yoe = st.slider("Years of Experience", 0, 50, 0)
yac = st.slider("Years at Company", 0, 50, 0)
female = st.checkbox("Female")
prediction_button = st.button("Predict")

if prediction_button:
    response = rq.post(
            'http://adrianalan.pythonanywhere.com/predict',
            json={"location": location,
                  "yoe": yoe,
                  "yac": yac,
                  "female": female}
            )
    prediction = json.loads(response.content)
    st.write("Your expected target salary is ${:,.0f}.".format(prediction['expected']*1000))
    st.write("Your low quantile is ${:,.0f}".format(prediction['quantile_low']*1000))
    st.write("Your high quantile is ${:,.0f}.".format(prediction['quantile_high']*1000))
   
    exp = prediction['expected']*1000
    qlow = prediction['quantile_low']*1000
    qhigh = prediction['quantile_high']*1000
    x = np.linspace(qlow-20000, qhigh+20000, 100)

    y = np.linspace(0.5, 0.5, 100)
    cols = np.linspace(0, 1, len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(10, 1))
    fig.set_facecolor((14./255, 17./255, 23./255))
    ax.set_facecolor((14./255, 17./255, 23./255))
    ax.get_yaxis().set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.tick_params(axis='x', colors='white', which='both')
    plt.xlim(qlow-20000, qhigh+20000)
    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(cols)
    lc.set_linewidth(20)
    line = ax.add_collection(lc)

    # Draw a horizontal scale
    plt.hlines(0, 0, 100)
    plt.eventplot([qlow, ghigh], orientation='horizontal', colors='w')
    plt.text(qlow, 1.1, '25% ', ha='right', va='center', color='w')
    plt.text(qhigh, 1.1, '75% ', ha='right', va='center', color='w')
   
    st.pyplot(fig)
