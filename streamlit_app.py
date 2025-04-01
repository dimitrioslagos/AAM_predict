import streamlit as st
import streamlit.components.v1
import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import predict_oil_future, html_future_oil_temp_plot, train_models_current, \
    compute_warning_on_bushing, compute_warning_on_DGA, display_light, generate_training_data_oil, \
    prepare_model_top_oil, predict_top_oil, html_error_plot

if 'OLMS_DATA' not in st.session_state:
    st.session_state.OLMS_DATA = None

if 'OLMS_DATA_top_oil_mapping' not in st.session_state:
    st.session_state.OLMS_DATA_top_oil_mapping = {'Top Oil Temperature': 'Top Oil Temp', 'Ambient Temperature': 'Ampient Sun', 'Ambient Shade Temperature': 'Ampient Shade', 'HV Current': 'HV Load Current'}

if 'Loading_mapping' not in st.session_state:
    st.session_state.Loading_mapping = {'Ambient Temperature': 'Ampient Sun', 'Ambient Shade Temperature': 'Ampient Sun', 'HV Current': None}

if 'DGA_mapping' not in st.session_state:
    st.session_state.DGA_mapping = {'H2': "TM8 0 H2inOil", 'CH4': "TM8 0 CH4inOil", 'C2H2': "TM8 0 C2H2inOil", 'C2H6': "TM8 0 C2H6", 'C2H4': "TM8 0 C2H4"}

if 'Bushings_mapping' not in st.session_state:
    st.session_state.Bushings_mapping = {'Cap H1': 'BUSHING H1 Capacitance', 'Cap H2': 'BUSHING H2 Capacitance', 'Cap H3': 'BUSHING H3 Capacitance',
                                         'Cap Y1': 'BUSHING Y1 Capacitance',
                                         'Cap Y2': 'BUSHING Y2 Capacitance',
                                         'Cap Y3': 'BUSHING Y3 Capacitance', 'tand H1': 'BUSHING H1 Tan delta', 'tand H2': 'BUSHING H2 Tan delta', 'tand H3': 'BUSHING H3 Tan delta',
                                         'tand Y1':  'BUSHING Y1 Tan delta', 'tand Y2': 'BUSHING Y2 Tan delta', 'tand Y3': 'BUSHING Y3 Tan delta'}

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

if "button1_pressed" not in st.session_state:
    st.session_state.button1_pressed = False

if "button2_pressed" not in st.session_state:
    st.session_state.button2_pressed = False

if "button3_pressed" not in st.session_state:
    st.session_state.button3_pressed = False

if "both_models_trained" not in st.session_state:
    st.session_state.both_models_trained = False

if "temperature_oil_predicted" not in st.session_state:
    st.session_state.temperature_oil_predicted = False

if 'OIL_temp' not in st.session_state:
    st.session_state.OIL_temp = None

if 'Probs' not in st.session_state:
    st.session_state.Probs = None

if 'mapping_confirmed' not in st.session_state:
    st.session_state.mapping_confirmed = False

# Set the title of the Streamlit app
st.title("Short Term Asset Management")

# Define the tabs
tabs = st.tabs(["Historical Data Input", "Alarms", "Predictions"])

# Content for the 'Home' tab

with tabs[0]:
    # Define the tabs
    tabs_historical = st.tabs(["File Upload", "Measurement Mapping", "Bushings Mapping", "DGA mapping"])
    with tabs_historical[0]:
        st.header("Historical Data Input")
        st.write("Provide Historical Data")

        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=["csv"], key=1)

        # Process the file after it's uploaded
        if uploaded_file is not None:
            if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != uploaded_file.name:
                # Store the uploaded file name in session state

                # Read the CSV file and generate curves
                if uploaded_file.name.endswith("csv"):
                    OLMS_DATA = pd.read_csv(uploaded_file, delimiter=';')
                    st.session_state['uploaded_file'] = uploaded_file.name
                    st.write("File content as DataFrame:")

                    # Prepare Data
                    id1 = OLMS_DATA.Timestamp.str.contains('EET')
                    id2 = OLMS_DATA.Timestamp.str.contains('EEST')
                    T1 = pd.to_datetime(OLMS_DATA.loc[id1, 'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
                    T2 = pd.to_datetime(OLMS_DATA.loc[id2, 'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
                    T = pd.concat((T1, T2))
                    OLMS_DATA.index = T
                    OLMS_DATA.drop(OLMS_DATA.index[OLMS_DATA.Logs == 'SENSOR ERROR 1'], inplace=True)
                    OLMS_DATA.drop(columns=['Logs'], inplace=True)
                    st.write(OLMS_DATA)
                    st.session_state['OLMS_DATA'] = OLMS_DATA
                    st.session_state.file_uploaded = True  #boolean to track that the right file has been uploaded
                    st.write("File has been uploaded successfully")
                else:
                    st.write("File is not csv")
            else:
                # If file is already uploaded, display the previous result from session state
                OLMS_DATA = st.session_state.get('OLMS_DATA', None)
        else:
            st.write("Please upload a file to see the content.")

        confirm_button4 = st.button("Confirm Measurements Mapping", key="Yes")  # No need for a key since it's a single button
        if confirm_button4:
            st.session_state.mapping_confirmed = True
            st.write("Mapping confirmed! Proceed to model training...")

        with tabs_historical[1]:
            if st.session_state.OLMS_DATA is not None:
                selected_options = {}
                st.write(st.session_state.OLMS_DATA_top_oil_mapping)
                for key in st.session_state.OLMS_DATA_top_oil_mapping.keys():
                    option = st.selectbox(
                        key + ":",
                        st.session_state.OLMS_DATA.Measurement.unique().tolist(),
                        key=f"select_{key}", index=None  # Unique key for each selectbox
                    )
                    st.write("You selected:", option)
                    selected_options[key] = option  # Store in new variable
                confirm_button1 = st.button("Confirm Choices", key="Measurement Mapping")  # No need for a key since it's a single button
                print(confirm_button1)
                if confirm_button1:
                    st.session_state.button1_pressed = True
                    st.session_state.OLMS_DATA_top_oil_mapping = selected_options
                    st.write("Selections confirmed:")
                    for key, value in selected_options.items():
                        st.write(f"**{key}:** {value}")

        with tabs_historical[2]:
            if st.session_state.OLMS_DATA is not None:
                # Do the mapping:
                selected_options = {}
                st.write(st.session_state.Bushings_mapping)
                for key in st.session_state.Bushings_mapping.keys():
                    option = st.selectbox(
                        key + ":",
                        st.session_state.OLMS_DATA.Measurement.unique().tolist(),
                        key=f"select_bushing_{key}"  # Unique key for each selectbox
                    )
                    st.write("You selected:", option)
                    st.session_state.Bushings_mapping[key] = option
                    selected_options[key] = option  # Store in new variable
                confirm_button2 = st.button("Confirm Choices", key="Bushings Mapping")  # No need for a key since it's a single button
                if confirm_button2:
                    st.session_state.button2_pressed = True
                    st.write("Selections confirmed:")
                    for key, value in selected_options.items():
                        st.write(f"**{key}:** {value}")

        with tabs_historical[3]:
            if st.session_state.OLMS_DATA is not None:
                # Do the mapping:
                selected_options = {}
                st.write(st.session_state.DGA_mapping)
                for key in st.session_state.DGA_mapping.keys():
                    option = st.selectbox(
                        key + ":",
                        st.session_state.OLMS_DATA.Measurement.unique().tolist(),
                        key=f"select_dga_{key}"  # Unique key for each selectbox
                    )
                    st.write("You selected:", option)
                    selected_options[key] = option  # Store in new variable
                confirm_button3 = st.button("Confirm Choices", key="DGA Mapping")  # No need for a key since it's a single button
                if confirm_button3:
                    st.session_state.DGA_mapping = option
                    st.write("Selections confirmed:")
                    for key, value in selected_options.items():
                        st.write(f"**{key}:** {value}")


    # train Oil temperature prediction model
    if st.session_state.mapping_confirmed and st.session_state.both_models_trained is False:  #this code runs every time the measurements mapping button is pressed
        if ('OLMS_DATA' in st.session_state) & ('model_oil' not in st.session_state):
            st.write(st.session_state.DGA_mapping)
            X, Y = generate_training_data_oil(OLMS_DATA, st.session_state.OLMS_DATA_top_oil_mapping, st.session_state.DGA_mapping)
            st.write(X)
            st.write(Y)
            st.write('Training Oil Temperature Prediction Model...')
            model_oil, oil_threshold = prepare_model_top_oil(X, Y)
            st.write('Oil Temperature Prediction Model trained')
            st.session_state['model_oil'] = model_oil
            st.session_state['oil_threshold'] = oil_threshold
            # Only for GA presentation
            st.session_state['X'] = X
            st.session_state['Y'] = Y
        else:
            st.write('Oil Temperature Prediction Model already trained')
            oil_threshold = st.session_state.get('oil_threshold', None)
            model_oil = st.session_state.get('model_oil', None)
            # Only for GA presentation
            X = st.session_state.get('X', None)
            Y = st.session_state.get('Y', None)
            ##train Oil temperature prediction model
        if ('OLMS_DATA' in st.session_state) and ('current_models' not in st.session_state):
            current_models = train_models_current(OLMS_DATA, st.session_state.OLMS_DATA_top_oil_mapping, horizon=6)
            st.write('Training Current Prediction Model...')
            model_oil, oil_threshold = prepare_model_top_oil(X, Y)
            st.write('Current Prediction Model trained')
            st.session_state['current_models'] = current_models
            # Only for GA presentation
            st.session_state['X'] = X
            st.session_state['Y'] = Y
        else:
            st.write('Current Prediction Model already trained')
            current_models = st.session_state.get('current_models', None)
        st.session_state.both_models_trained = True

    elif 'X' in st.session_state and 'Y' in st.session_state:
        # If models are already trained, just use the existing X, Y data
        X = st.session_state.get('X', None)
        Y = st.session_state.get('Y', None)
        model_oil = st.session_state.get('model_oil', None)
        oil_threshold = st.session_state.get('oil_threshold', None)
        current_models = st.session_state.get('current_models', None)
    else:
        # If no model has been trained, show an error or a message
        st.write('Models not trained yet')

with tabs[1]:
    st.header("Real Time Alarms")
    col1, col2, col3 = st.columns([20, 20, 20])
    if st.session_state.file_uploaded:
        with col1:
            if st.session_state.button2_pressed:
                st.subheader("Bushings")
                if 'OLMS_DATA' in st.session_state:
                    Alarms = compute_warning_on_bushing(pd.to_datetime('2024-08-01'), OLMS_DATA, st.session_state.Bushings_mapping)
                    st.markdown(display_light(Alarms.empty), unsafe_allow_html=True)
                    if not ((Alarms.loc['Status'] == 'Warning').any()):
                        st.write("No warnings on the Bushings. Bushing status ok")
                    else:
                        st.write("Warnings on the Bushings")
                    st.write("Bushings condition:")
                    st.write(Alarms)
        with col2:
            st.subheader("Dissolved Gas Analysis")
            if 'OLMS_DATA' in st.session_state and st.session_state.button3_pressed:
                DGA = compute_warning_on_DGA(pd.to_datetime('2024-08-01'), OLMS_DATA, st.session_state.DGA_mapping)
                st.markdown(display_light(not ((DGA['SCORE'] <= 3).any())), unsafe_allow_html=True)
                if (DGA['SCORE'] <= 3).any():
                    DGA['SCORE'] = 'ok'
                    st.write("Status ok. No action suggested based on DGA data")
                elif (3 < DGA['SCORE']).any() & (DGA['SCORE'] <= 5).any():
                    DGA['SCORE'] = 'minor warning'
                    st.write("Gas levels considerable but within limits. Increase monitoring")
                elif (5 < DGA['SCORE']).any() & (DGA['SCORE'] <= 7).any():
                    DGA['SCORE'] = 'warning'
                    st.write("Gas levels considerable. Maintenance actions suggested")
                else:
                    DGA['SCORE'] = 'critical'
                    st.write("Gas levels critical. Maintenance actions suggested")
                st.write("DGA Results")
                st.write(DGA)
        with col3:
            st.subheader("Oil Anomaly Detection")
            if 'model_oil' in st.session_state :
                t = pd.to_datetime('2024-08-01')
                Xtest = X[(X.index >= (t - pd.Timedelta(days=1))) & (X.index <= t)]
                Ytest = Y[(Y.index >= (t - pd.Timedelta(days=1))) & (Y.index <= t)]
                Flags, Error = predict_top_oil(Xtest, Ytest, model_oil, oil_threshold)
                st.markdown(display_light(((Flags == True).any())), unsafe_allow_html=True)
                if not ((Flags == True).any()):
                    st.write("No anomalies detected in oil temperature")
                else:
                    st.write("Anomalies detected in oil temperature")
                st.write('Model Output past 24 hours')
                st.components.v1.html(html_error_plot(Error, oil_threshold), height=500)
with tabs[2]:
    if st.session_state.button1_pressed and st.session_state.button3_pressed and st.session_state.both_models_trained:
        st.subheader("Oil Temperature Prediction")
        if ('model_oil' in st.session_state) & ('current_models' in st.session_state):
            t = pd.to_datetime('2024-06-13 11:00:00')
            if st.session_state.temperature_oil_predicted is False:
                st.session_state.OIL_temp, st.session_state.Probs = predict_oil_future(model_oil, current_models, OLMS_DATA, st.session_state.OLMS_DATA_top_oil_mapping, st.session_state.Loading_mapping, st.session_state.DGA_mapping, t)
                st.session_state.temperature_oil_predicted = True
            st.write('Failure Probability due to oil Temperatures')
            st.write(st.session_state.Probs * 0)
            st.components.v1.html(html_future_oil_temp_plot(st.session_state.OIL_temp), height=600)


