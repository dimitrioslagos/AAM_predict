import streamlit as st
import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import predict_oil_future, html_future_oil_temp_plot, train_models_current,compute_warning_on_bushing, compute_warning_on_DGA,display_light,generate_training_data_oil,prepare_model_top_oil, predict_top_oil, html_error_plot



# Set the title of the Streamlit app
st.title("Short Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input","Alarms","Predictions"])

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
    
    ##train Oil temperature prediction model
    if ('OLMS_DATA'  in st.session_state)&('model_oil' not in st.session_state):  
        X, Y = generate_training_data_oil(OLMS_DATA)
        st.write('Training Oil Temperature Prediction Model...')
        model_oil,oil_threshold = prepare_model_top_oil(X,Y)
        st.write('Oil Temperature Prediction Model trained')
        st.session_state['model_oil'] = model_oil
        st.session_state['oil_threshold'] = oil_threshold
        #Only for GA presentation
        st.session_state['X'] = X
        st.session_state['Y'] = Y
    else:
        st.write('Oil Temperature Prediction Model not trained')
        oil_threshold = st.session_state.get('oil_threshold', None)
        model_oil = st.session_state.get('model_oil', None)
        #Only for GA presentation
        X = st.session_state.get('X', None)
        Y = st.session_state.get('Y', None)
        ##train Oil temperature prediction model
    if ('OLMS_DATA' in st.session_state) & ('current_models' not in st.session_state):
        current_models = train_models_current(OLMS_DATA, horizon=6)
        st.write('Training Current Prediction Model...')
        model_oil, oil_threshold = prepare_model_top_oil(X, Y)
        st.write('Current Prediction Model trained')
        st.session_state['current_models'] = current_models
        # Only for GA presentation
        st.session_state['X'] = X
        st.session_state['Y'] = Y
    else:
        st.write('Current Prediction Model not trained')
        current_models = st.session_state.get('current_models', None)


with tabs[1]:
    st.header("Real Time Alarms")
    col1, col2, col3 = st.columns([20,20,20])
    with col1:
        st.subheader("Bushings")
        if 'OLMS_DATA'  in st.session_state:
            Alarms = compute_warning_on_bushing(pd.to_datetime('2024-08-01'),OLMS_DATA)
            st.markdown(display_light(Alarms.empty), unsafe_allow_html=True)
            if not((Alarms.loc['Status'] == 'Warning').any()):
                st.write("No warnings on the Bushings. Bushing status ok")
            else:
                st.write("Warnings on the Bushings")
            st.write("Bushings condition:")
            st.write(Alarms)
    with col2:
        st.subheader("Dissolved Gas Analysis")
        if 'OLMS_DATA'  in st.session_state:
            DGA = compute_warning_on_DGA(pd.to_datetime('2024-08-01'),OLMS_DATA)
            st.markdown(display_light(not((DGA['SCORE']<=3).any())), unsafe_allow_html=True)
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
    with col3:
        st.subheader("Oil Anomaly Detection")
        if ('model_oil' in st.session_state):
            t = pd.to_datetime('2024-08-01')
            Xtest = X[(X.index >= (t - pd.Timedelta(days=1))) & (X.index <= t)]
            Ytest = Y[(Y.index >= (t - pd.Timedelta(days=1))) & (Y.index <= t)]
            Flags,Error = predict_top_oil(Xtest,Ytest,model_oil,oil_threshold)
            st.markdown(display_light(((Flags==True).any())), unsafe_allow_html=True)
            if not((Flags==True).any()):
                st.write("No anomalies detected in oil temperature")
            else:
                st.write("Anomalies detected in oil temperature")
            st.write('Model Output past 24 hours')
            st.components.v1.html(html_error_plot(Error,oil_threshold), height=500)
    
with tabs[2]:
    st.subheader("Oil Temperature Prediction")
    if ('model_oil' in st.session_state)&('current_models' in st.session_state):
        t = pd.to_datetime('2024-06-13 11:00:00')
        OIL_temp, Probs = predict_oil_future(model_oil, current_models, OLMS_DATA, t)
        st.write('Failure Probability due to oil Temperatures')
        st.write(Probs*0)
        st.components.v1.html(html_future_oil_temp_plot(OIL_temp),height=600)

