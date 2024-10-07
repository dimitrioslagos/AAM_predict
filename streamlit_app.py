import streamlit as st
import pandas as pd
import math
from pathlib import Path



# Set the title of the Streamlit app
st.title("Short Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input"])

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
                st.write(OLMS_DATA )

                #Prepare Data
                id1 = OLMS_DATA.Timestamp.str.contains('EET')
                id2 = OLMS_DATA.Timestamp.str.contains('EEST')
                T1 = pd.to_datetime(OLMS_DATA.loc[id1,'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
                T2 = pd.to_datetime(OLMS_DATA.loc[id2,'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
                T = pd.concat((T1,T2))
                OLMS_DATA.index = T
                OLMS_DATA.drop(OLMS_DATA.index[OLMS_DATA.Logs=='SENSOR ERROR 1'],inplace=True)
                OLMS_DATA.drop(columns=['Logs'],inplace=True)
                st.session_state['OLMS_DATA'] = OLMS_DATA
            else:
                st.write("File is not csv")
        else:
            # If file is already uploaded, display the previous result from session state
            OLMS_DATA = st.session_state.get('OLMS_DATA', None)
    else:
        st.write("Please upload a file to see the content.")
