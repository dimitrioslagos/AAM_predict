import streamlit as st
import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import compute_warning_on_bushing


# Set the title of the Streamlit app
st.title("Short Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input","Alarms"])

# Content for the 'Home' tab
with tabs[0]:
    st.header("Historical Data Input")
    st.write("Provide Historical Data (OLMS)")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv"],key=1)

    # Process the file after it's uploaded
    if uploaded_file is not None:
        if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != uploaded_file.name:
            # Store the uploaded file name in session state

            # Read the CSV file and generate curves
            if uploaded_file.name.endswith("csv"):
                OLMS_DATA = pd.read_csv(uploaded_file,delimiter=';')
                st.session_state['uploaded_file'] = uploaded_file.name
                st.write("File content as DataFrame:")

                #Prepare Data
                id1 = OLMS_DATA.Timestamp.str.contains('EET')
                id2 = OLMS_DATA.Timestamp.str.contains('EEST')
                T1 = pd.to_datetime(OLMS_DATA.loc[id1,'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
                T2 = pd.to_datetime(OLMS_DATA.loc[id2,'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
                T = pd.concat((T1,T2))
                OLMS_DATA.index = T
                OLMS_DATA.drop(OLMS_DATA.index[OLMS_DATA.Logs=='SENSOR ERROR 1'],inplace=True)
                OLMS_DATA.drop(columns=['Logs'],inplace=True)
                st.write(OLMS_DATA )
                st.session_state['OLMS_DATA'] = OLMS_DATA
            else:
                st.write("File is not csv")
        else:
            # If file is already uploaded, display the previous result from session state
            OLMS_DATA = st.session_state.get('OLMS_DATA', None)
    else:
        st.write("Please upload a file to see the content.")

with tabs[1]:
    st.header("Real Time Alarms")
    st.subheader("Bushings")
    if 'OLMS_DATA'  in st.session_state:
        Alarms = compute_warning_on_bushing(pd.to_datetime('2024-08-01'),OLMS_DATA)
        if Alarms.empty:
            st.write("No warnings on the Bushings. Bushing status ok")
            st.write(Alarms)
        else:
            st.write("Warnings on the Bushings")
        st.write(Alarms)
    st.subheader("Dissolved Gas Analysis")
    if 'OLMS_DATA'  in st.session_state:
        DGA = ccompute_warning_on_DGA(pd.to_datetime('2024-08-01'),OLMS_DATA)
        if DGA['Score'] <= 3:
            return st.write("Status ok. No action suggested based on DGA data")
        elif 3 < c2h6_value <= 5:
            return st.write("Gas levels considerable but within limits. Increase monitoring")
        elif 5 < c2h6_value <= 7:
            return st.write("Gas levels considerable. Maintenance actions suggested")
        else:
            return st.write("Gas levels critical. Maintenance actions suggested") 
        st.write("DGA Results")
        st.write(DGA)

            
    
