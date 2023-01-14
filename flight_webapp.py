import streamlit as st
import pickle
import numpy as np

lin=pickle.load(open('lin_model.pkl','rb'))
dt = pickle.load(open('dec_tree_model.pkl','rb'))
rf= pickle.load(open('rand_for_model.pkl','rb'))
xgb = pickle.load(open('xgb_model.pkl','rb'))
ada = pickle.load(open('ada_reg_model.pkl','rb'))

st.title("Flight Price Prediction Web App")
html_temp = """
    <div style="background-color:lightgreen ;padding:8px">
    <h2 style="color:black;text-align:center;">Flight Price Prediction</h2>
    </div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
activities = ['Linear Regression','Decision Tree','Random Forest','AdaBoost']
option = st.sidebar.selectbox('Which regression model would you like to use?',activities)
st.subheader(option)

Airlines = [0,1,2,3,4,5]
airline_option = st.sidebar.selectbox("Choose the Airlines",Airlines)
airline_option=float(airline_option)

sr_city = [0,1,2,3,4,5]
sr_city_option = st.sidebar.selectbox("Choose the Source city",sr_city)
sr_city_option=float(sr_city_option)

d_time = [0,1,2,3,4,5]
d_time_option = st.sidebar.selectbox("Choose the departure time",d_time)
d_time_option=float(d_time_option)

stops = [0,1,2]
stops_option = st.sidebar.selectbox("Choose the number of stops in between",stops)
stops_option=float(stops_option)

a_time = [0,1,2,3,4,5]
a_time_option = st.sidebar.selectbox("Choose the arrival time",a_time)
a_time_option=float(a_time_option)

d_city = [0,1,2,3,4,5]
d_city_option = st.sidebar.selectbox("Choose the Destination city",d_city)
d_city_option=float(d_city_option)

classs = [0,1]
class_option = st.sidebar.selectbox("Choose the class",classs)
class_option=float(class_option)

duration = st.slider('Select duration of flight', 0, 28)
days_left = st.slider('Select days left', 0, 60)

inputs=[[airline_option,sr_city_option,d_time_option,stops_option,a_time_option,d_city_option,class_option,
duration,days_left]]

if st.button('Predict'):
    if option=='Linear Regression':
        st.success(lin.predict(inputs))
    elif option=='Decision Tree':
        st.success(dt.predict(inputs))
    elif option=='Random Forest':
        st.success(rf.predict(inputs))
    # elif option=='XGBoost':
    #     st.success(xgb.predict(inputs))
    else:
        st.success(ada.predict(inputs))
