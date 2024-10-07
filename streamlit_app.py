import streamlit as st
import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import compute_warning_on_bushing, compute_warning_on_DGA,display_light


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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Bushings")
        if 'OLMS_DATA'  in st.session_state:
            Alarms = compute_warning_on_bushing(pd.to_datetime('2024-08-01'),OLMS_DATA)
            st.markdown(display_light(Alarms.empty), unsafe_allow_html=True)
            if Alarms.empty:
                st.write("No warnings on the Bushings. Bushing status ok")
                st.write(Alarms)
            else:
                st.write("Warnings on the Bushings")
            st.write(Alarms)
    with col2:
        st.subheader("Dissolved Gas Analysis")
        if 'OLMS_DATA'  in st.session_state:
            DGA = compute_warning_on_DGA(pd.to_datetime('2024-08-01'),OLMS_DATA)
            st.markdown(display_light((DGA['SCORE']<=3).any()), unsafe_allow_html=True)
            if (DGA['SCORE']<=3).any():
                DGA['SCORE'] = 'ok'
                st.write("Status ok. No action suggested based on DGA data")
            elif (3 < DGA['SCORE']).any() & (DGA['SCORE']<=5).any():
                DGA['SCORE'] = 'minor warning'
                st.write("Gas levels considerable but within limits. Increase monitoring")
            elif (5 < DGA['SCORE']).any() & (DGA['SCORE']<=7).any():
                DGA['SCORE'] = 'warning'
                st.write("Gas levels considerable. Maintenance actions suggested")
            else:
                DGA['SCORE'] = 'critical'
                st.write("Gas levels critical. Maintenance actions suggested") 
            st.write("DGA Results")
            st.write(DGA)


    
