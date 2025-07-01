import streamlit as st
import streamlit.components.v1
import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import  html_future_oil_temp_plot, train_q90_model, \
     compute_warning_on_DGA, display_light, generate_training_data_oil, \
    prepare_model_top_oil, predict_top_oil, html_error_plot, train_mean_model, predict_q90_model, probability_to_exceed
from sqlalchemy import create_engine
import numpy as np
import os
from datetime import datetime, timedelta, time
import json
import keras


user = "'IPTO'"
#Create Engine
server = '147.102.30.47'            # or IP address / named instance
database = 'opentunity_dev'
username = 'opentunity'
password = '0pentunity44$$'
conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(conn_str)

def is_model_stale(metadata_path, days=7):

    if not os.path.isfile(metadata_path):
        return True  # No model or metadata found → needs retraining

    with open(metadata_path) as f:
        metadata = json.load(f)

    trained_on = datetime.strptime(metadata["trained_on"], "%Y-%m-%d_%H-%M-%S")
    return datetime.now() - trained_on > timedelta(days=days)

def is_directory_empty(path):
    return not os.listdir(path)


def write_measurements_in_db(engine,meas_IDs,asset_name):
    Data = st.session_state.OLMS_DATA
    asset_ID = get_asset_ID(engine,asset_name)
    all_data = []
    for measurement in Data.Measurement.unique():
        temp_df = Data[Data.Measurement == measurement].copy()
        if temp_df.empty:
            continue
        #To be altered according to the user country
        matching_keys1 = [k for k, v in st.session_state.Bushings_mapping.items() if v == measurement]
        matching_keys2 = [k for k, v in st.session_state.OLMS_DATA_top_oil_mapping.items() if v == measurement]
        matching_keys3 = [k for k, v in st.session_state.DGA_mapping.items() if v == measurement]
        matching_key= matching_keys1+matching_keys2+matching_keys3
        if len(matching_key)==1:
            df_chunk = pd.DataFrame({
                        'Timestamp': temp_df.index.tz_localize('Europe/Athens', ambiguous='NaT').tz_convert('UTC').tz_localize(None),  # Remove timezone info for SQL
                        'Value': temp_df.Value.astype(float),
                        'AssetID': [asset_ID] * len(temp_df),
                        'MeasurementID': [meas_IDs[meas_IDs.Name == matching_key[0]].MeasurementID.values[0]] * len(temp_df)
                    })
            print(df_chunk)


            existing_ts = pd.read_sql_query(
                    f"""
                    SELECT Timestamp FROM Measurements
                    WHERE MeasurementID = {meas_IDs[meas_IDs.Name == matching_key[0]].MeasurementID.values[0]} AND AssetID = {asset_ID}
                    AND Timestamp BETWEEN '{df_chunk['Timestamp'].min()}' AND '{df_chunk['Timestamp'].max()}'
                    """,
                    con=engine
                )
            df_chunk = df_chunk[~df_chunk['Timestamp'].isin(existing_ts['Timestamp'])]
            all_data.append(df_chunk)
        #
     # # Combine everything
    final_df = pd.concat(all_data, ignore_index=True)
    final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')
    final_df = final_df[np.isfinite(final_df['Value'])]
    print(final_df)
    #
    # Drop invalid timestamps (just in case)
    final_df = final_df.dropna(subset=['Timestamp'])

    # Insert once
    final_df.to_sql('Measurements', con=engine, if_exists='append', index=False, chunksize=50000)

def get_measurements_IDs(engine):

    measurement_IDS = pd.read_sql("SELECT MeasurementID,Name FROM MeasurementTypes", con=engine)
    return measurement_IDS

def get_asset_ID(engine,asset_name):
    asset_ID = pd.read_sql("SELECT AssetID FROM Assets where AssetName="+"'"+asset_name+"'", con=engine)
    return asset_ID.values[0][0]

def check_file(uploaded_file):
    try:
        # Try parsing the file and Timestamp column
        OLMS_DATA = pd.read_csv(
                uploaded_file,
                delimiter=';',
                parse_dates=['Timestamp'],
                dayfirst=False,  # or True if your dates are D/M/Y
                low_memory=False
            )

        # Validate required columns exist
        required_columns = {'Timestamp', 'Measurement', 'Value'}
        if not required_columns.issubset(OLMS_DATA.columns):
            st.error("❌ CSV must contain columns: Timestamp, Measurement, Value")
        else:
            # Validate column types
            is_valid = True
            errors = []

        # Check Timestamp parsed correctly
        if not pd.api.types.is_datetime64_any_dtype(OLMS_DATA['Timestamp']):
            is_valid = False
            errors.append("Timestamp column is not a valid datetime.")

        # Check Measurement is string (object)
        if not pd.api.types.is_object_dtype(OLMS_DATA['Measurement']):
            is_valid = False
            errors.append("Measurement column must be of string type.")

        # Check Value is float-compatible
        try:
            OLMS_DATA['Value'] = OLMS_DATA['Value'].astype(float)
        except ValueError:
            is_valid = False
            errors.append("Value column must contain only numeric (float) values.")

        if is_valid:
            st.success("✅ File format is valid!")
            st.session_state['uploaded_file'] = uploaded_file.name
                # Proceed with rest of the app using OLMS_DATA
        else:
            for err in errors:
                st.error(f"❌ {err}")

    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")

def import_data(assets,measurements_IDs):
    # Define the tabs
    tabs_historical = st.tabs(["File Upload"])
    with tabs_historical[0]:
        st.header("Historical Data Input")
        st.write("Provide Historical Data")

        st.session_state.asset = st.selectbox("Asset:",
                                       assets.AssetName,
                                        key=f"select_asset", index=None  # Unique key for each selectbox
                                    )
        if st.session_state.asset is not None:
            # File uploader
            uploaded_file = st.file_uploader("Choose a file", type=["csv"], key=1)

            # Process the file after it's uploaded
            if (uploaded_file is not None)&(st.session_state.asset is not None):
                if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] != uploaded_file.name:
                    if uploaded_file is not None:
                        st.session_state['uploaded_file'] = uploaded_file.name
                        REQUIRED_COLUMNS = ['Timestamp', 'Measurement', 'Value']
                        try:
                            OLMS_DATA = pd.read_csv(
                                uploaded_file,
                                delimiter=';',
                                parse_dates=['Timestamp'],
                                low_memory=False
                            )

                            # Check for required columns
                            missing_cols = [col for col in REQUIRED_COLUMNS if col not in OLMS_DATA.columns]
                            if missing_cols:
                                st.error(f"❌ Missing required columns: {missing_cols}")
                                st.stop()

                            # Set index
                            OLMS_DATA.set_index('Timestamp', inplace=True)

                            # Check data types if needed (optional)
                            if not pd.api.types.is_numeric_dtype(OLMS_DATA['Value']):
                                st.error("❌ 'Value' column must be numeric.")
                                st.stop()

                            if not pd.api.types.is_string_dtype(OLMS_DATA['Measurement']):
                                st.error("❌ 'Value' must be string.")
                                st.stop()

                            # Success
                            st.write("✅ File loaded and indexed successfully")
                            st.write("File content as DataFrame:")
                            st.write(OLMS_DATA.head())
                            st.session_state['OLMS_DATA'] = OLMS_DATA
                            st.session_state.file_uploaded = True
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error processing file: {e}")

                        st.write("✅ File loaded and indexed successfully")
                        st.write("File content as DataFrame:")
                        st.write(OLMS_DATA.head())
                        st.session_state['OLMS_DATA'] = OLMS_DATA
                        st.session_state.file_uploaded = True  #boolean to track that the right file has been uploaded
                        st.write("File has been uploaded successfully")
                        st.rerun()
                    else:
                        st.write("File is not csv")
                else:
                    # If file is already uploaded, display the previous result from session state
                    if (st.session_state.OLMS_DATA is not None)&(not(st.session_state.mapping_confirmed)):
                        with st.expander("Measurement Mapping", expanded=True):
                            tabs_mapping = st.tabs(["Measurement Mapping", "Bushings Mapping", "DGA mapping"])
                            with tabs_mapping[0]:
                                if st.session_state.OLMS_DATA is not None:
                                    selected_options = {}
                                    st.write(st.session_state.OLMS_DATA_top_oil_mapping)
                                    for key in st.session_state.OLMS_DATA_top_oil_mapping.keys():
                                        option = st.selectbox(
                                            key + ":",
                                            st.session_state.OLMS_DATA.Measurement.unique().tolist()+[None],
                                            key=f"select_{key}", index=None  # Unique key for each selectbox
                                        )
                                        st.write("You selected:", option)
                                        selected_options[key] = option  # Store in new variable
                                    confirm_button1 = st.button("Confirm Choices", key="Measurement Mapping")  # No need for a key since it's a single button
                                    if confirm_button1:
                                        st.session_state.OLMS_DATA_top_oil_mapping = selected_options
                                        st.write("Selections confirmed:")
                                        for key, value in selected_options.items():
                                            st.write(f"**{key}:** {value}")
                                        st.rerun()
                            with tabs_mapping[1]:
                                selected_options = {}
                                st.write(st.session_state.Bushings_mapping)
                                for key in st.session_state.Bushings_mapping.keys():
                                    option = st.selectbox(
                                            key + ":",
                                            st.session_state.OLMS_DATA.Measurement.unique().tolist()+[None],
                                            key=f"select_bushing_{key}"  # Unique key for each selectbox
                                        )
                                    st.write("You selected:", option)
                                    selected_options[key] = option  # Store in new variable
                                confirm_button2 = st.button("Confirm Choices", key="Bushings Mapping")  # No need for a key since it's a single button
                                if confirm_button2:
                                    st.session_state.Bushings_mapping = selected_options
                                    st.write("Selections confirmed:")
                                    for key, value in selected_options.items():
                                        st.write(f"**{key}:** {value}")
                                    st.rerun()
                            with tabs_mapping[2]:
                                selected_options = {}
                                st.write(st.session_state.DGA_mapping)
                                for key in st.session_state.DGA_mapping.keys():
                                    option = st.selectbox(
                                            key + ":",
                                            st.session_state.OLMS_DATA.Measurement.unique().tolist()+[None],
                                            key=f"select_dga_{key}"  # Unique key for each selectbox
                                        )
                                    st.write("You selected:", option)
                                    selected_options[key] = option  # Store in new variable
                                confirm_button3 = st.button("Confirm Choices", key="DGA")  # No need for a key since it's a single button
                                if confirm_button3:
                                    st.session_state.DGA_mapping = selected_options
                                    st.write("Selections confirmed:")
                                    for key, value in selected_options.items():
                                        st.write(f"**{key}:** {value}")
                                    st.rerun()
            else:
                st.write("Please upload a file to see the content.")
            if not(st.session_state.mapping_confirmed):
                confirm_button4 = st.button("Confirm Measurements & Mapping", key="Yes")  # No need for a key since it's a single button
                if confirm_button4:
                    st.session_state.mapping_confirmed = True
                    st.write("Mapping confirmed! Updating Database...")
                    write_measurements_in_db(engine,measurements_IDs,st.session_state.asset)


def read_data_from_DB(asset_ID, engine):
    data = pd.read_sql_query( f"""
                SELECT * FROM Measurements
                WHERE AssetID = {asset_ID}
                """,
                con=engine)
    return data

def train_top_oil_anomaly_detection_model(Data,top_oil_mapping,DGA_mapping,Bushings_mapping):
    X, Y = generate_training_data_oil(Data,top_oil_mapping,DGA_mapping, Bushings_mapping)
    model_oil, oil_threshold = prepare_model_top_oil(X, Y)
    #Store model and metadata
    return model_oil, oil_threshold


def get_assets_of_user(user):
    query = f"SELECT * FROM assets WHERE Owner = {user} and Tool = 'ST_AAM'"

    # Read the result into a DataFrame
    user_assets = pd.read_sql(query, con=engine)
    return user_assets


def update_anomaly_detection_model(asset,measurements_IDs):
    Data = read_data_from_DB(asset['AssetID'], engine)
    for meas_id in Data.MeasurementID.unique().tolist():
        Data.loc[Data.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
    Data.index = pd.DatetimeIndex(Data.Timestamp)
    Data.drop(columns=['AssetID','Timestamp'],inplace=True)
    Data = Data.rename(columns={'MeasurementID': 'Measurement'})
    DGA_mapping = {'H2': "", 'CH4': "", 'C2H2': "", 'C2H6': "", 'C2H4': ""}
    for key in DGA_mapping.keys():
        if key in Data.Measurement.unique().tolist():
            DGA_mapping[key]=key
        else:
            DGA_mapping[key]=None

    Bushings_mapping = {'Capacitance HV1': '', 'Capacitance HV2': '',
                                'Capacitance HV3': '','Capacitance LV1': '',
                                'Capacitance LV2': '', 'Capacitance LV3': '',
                                'tand HV1': '', 'tand HV2': '', 'tand HV3': '',
                                             'tand LV1':  '', 'tand LV2': '', 'tand LV3': ''}
    for key in Bushings_mapping.keys():
        if key in Data.Measurement.unique().tolist():
            Bushings_mapping[key]=key
        else:
            Bushings_mapping[key]=None
    top_oil_mapping={'Top Oil Temperature': '', 'Ambient Temperature': '',
                             'HV Current': ''}
    for key in top_oil_mapping.keys():
        if key in Data.Measurement.unique().tolist():
            top_oil_mapping[key]=key
        else:
            top_oil_mapping[key]=None
        if any(value is None for value in top_oil_mapping.values()):
            continue
        model, threshold = train_top_oil_anomaly_detection_model(Data,top_oil_mapping,DGA_mapping,Bushings_mapping)
        # Save with timestamp
        train_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"models/top_oil_anomaly/{asset['AssetName']}.h5"
        model.save(model_dir)
        metadata = {
            "trained_on": train_date,
            "threshold": threshold,
                }
        with open(f"models/top_oil_anomaly/{asset['AssetName']}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)



def update_q90_model(asset,measurements_IDs):
    Data = read_data_from_DB(asset['AssetID'], engine)
    for meas_id in Data.MeasurementID.unique().tolist():
        Data.loc[Data.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
    Data.index = pd.DatetimeIndex(Data.Timestamp)
    Data.drop(columns=['AssetID','Timestamp'],inplace=True)
    Data = Data.rename(columns={'MeasurementID': 'Measurement'})
    if Data.shape[0]>=24*30*5:
        model = train_q90_model(Data)
        train_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"models/oil_temp_forecast/{asset['AssetName']}_90.keras"
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        model.save(model_dir)
        metadata = {
                "trained_on": train_date,
                    }
        with open(f"models/oil_temp_forecast/{asset['AssetName']}_90_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


def update_q50_model(asset,measurements_IDs):
    Data = read_data_from_DB(asset['AssetID'], engine)
    for meas_id in Data.MeasurementID.unique().tolist():
        Data.loc[Data.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
    Data.index = pd.DatetimeIndex(Data.Timestamp)
    Data.drop(columns=['AssetID','Timestamp'],inplace=True)
    Data = Data.rename(columns={'MeasurementID': 'Measurement'})
    if Data.shape[0]>=24*30*5:
        model = train_mean_model(Data)
        train_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = f"models/oil_temp_forecast/{asset['AssetName']}_50.keras"
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        model.save(model_dir)
        metadata = {
                "trained_on": train_date,
                    }
        with open(f"models/oil_temp_forecast/{asset['AssetName']}_50_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

def calculate_DGA_flag(Data,measurements_IDs):
    DGA_mapping = {'H2': "H2", 'CH4': "CH4", 'C2H2': "C2H2", 'C2H6': "C2H6", 'C2H4': "C2H4"}
    dga_ids = [meas['MeasurementID'] for id, meas in measurements_IDs.iterrows() if meas['Name'] in DGA_mapping.keys()]
    Data_DGA = Data[Data["MeasurementID"].isin(dga_ids)]
    if Data_DGA.shape[0]>=1:
        for meas_id in dga_ids:
            Data_DGA.loc[Data_DGA.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
        Data_DGA = Data_DGA.rename(columns={'MeasurementID': 'Measurement'})
        Data_DGA.index = pd.DatetimeIndex(Data_DGA.Timestamp)
        Data_DGA.drop(columns = ['Timestamp','AssetID'],inplace=True)
        DGA_flag = compute_warning_on_DGA(Data_DGA)
    else:
        DGA_flag = pd.DataFrame()
    return DGA_flag

def calculate_bushings_alers(Data,measurements_IDs):
    Bushings_mapping = {'Capacitance HV1': 'Capacitance HV1', 'Capacitance HV2': 'Capacitance HV2',
                                    'Capacitance HV3': 'Capacitance HV3','Capacitance LV1': 'Capacitance LV1',
                                    'Capacitance LV2': 'Capacitance LV2', 'Capacitance LV3': 'Capacitance LV3',
                                    'tand HV1': 'tand HV1', 'tand HV2': 'tand HV2', 'tand HV3': 'tand HV3',
                                                 'tand LV1':  'tand LV1', 'tand LV2': 'tand LV2', 'tand LV3': 'tand LV3'}
    Bushings_ids = [meas['MeasurementID'] for id, meas in measurements_IDs.iterrows() if meas['Name'] in Bushings_mapping.keys()]
    Data_Bushings = Data[Data["MeasurementID"].isin(Bushings_ids)]
    for meas_id in Bushings_ids:
        Data_Bushings.loc[Data_Bushings.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
    Data_Bushings = Data_Bushings.rename(columns={'MeasurementID': 'Measurement'})
    # Pivot the data
    Data_Bushings = Data_Bushings.pivot(index="Timestamp", columns="Measurement", values="Value")
    Data_Bushings = Data_Bushings.sort_index().sort_index(axis=1)
    Data_Bushings = Data_Bushings[(Data_Bushings != 0).all(axis=1)]
    df = Data_Bushings.copy()

    cap_cols = [col for col in df.columns if "Capacitance" in col]
    cap_df = df[cap_cols].copy()


    # Step 1: Calculate previous day's daily mean (used for comparison)
    daily_mean = cap_df.resample("D").mean()

    # Step 2: Resample into 2-hour means
    resampled_4h = cap_df.resample("4H").mean()

    # Step 3: Get previous day's mean for each 2-hour window
    # Create a column with the date of the *previous* day
    prev_day_index = resampled_4h.index - pd.Timedelta(days=1)
    prev_day_means = daily_mean.reindex(prev_day_index.date)

    # Because index mismatch on datetime vs. date, align it
    prev_day_means.index = resampled_4h.index  # force same shape for element-wise math

    # Step 4: Calculate percentage change
    pct_change = (resampled_4h - prev_day_means) / prev_day_means * 100

    # Step 5: Flag large deviations
    alerts = (pct_change < -10) | (pct_change > 5)

    # Optional: Get only the rows where any alert is triggered
    alerted_changes = pct_change[alerts.any(axis=1)]
    return alerted_changes

def calculate_top_oil_alarms(Data,measurements_IDs):
    oil_anomaly_detection_mapping = {'Top Oil Temperature': 'Top Oil Temperature', 'Ambient Temperature': 'Ambient Temperature',
                                          'HV Current': 'HV Current'}
    oil_ids = [meas['MeasurementID'] for id, meas in measurements_IDs.iterrows() if meas['Name'] in oil_anomaly_detection_mapping.keys()]
    Data_oil = Data[Data["MeasurementID"].isin(oil_ids)]
    for meas_id in oil_ids:
        Data_oil.loc[Data_oil.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
    Data_oil = Data_oil.rename(columns={'MeasurementID': 'Measurement'})
    Data_oil.index = pd.DatetimeIndex(Data_oil.Timestamp)
    Data_oil = Data_oil.pivot(index="Timestamp", columns="Measurement", values="Value")
    if Data_oil.shape[0]>=2:
        Data_oil = Data_oil.resample('60min').mean()
        model = keras.models.load_model('models/top_oil_anomaly/' + asset + '.h5', compile=False)
        with open('models/top_oil_anomaly/'+asset+'_metadata.json', 'r') as f:
            config = json.load(f)
        threshold = config['threshold']
        train_ids = Data_oil.index[Data_oil.rolling('60min').count().sum(axis=1) == Data_oil.shape[1]]
        train_ids = train_ids[1:]
        Y = Data_oil.loc[train_ids, 'Top Oil Temperature']
        X = Data_oil.shift(1).loc[train_ids]
        X= X.dropna(axis=0)
        Y = Y.loc[X.index]
        alarms_oil = predict_top_oil(X,Y,model, threshold)
    else:
        alarms_oil = pd.DataFrame()
    return alarms_oil,threshold,Data_oil

assets = get_assets_of_user(user)
measurements_IDs = get_measurements_IDs(engine)


st.set_page_config(layout='wide')
if 'OLMS_DATA' not in st.session_state:
    st.session_state.OLMS_DATA = None

if 'OLMS_DATA_top_oil_mapping' not in st.session_state:
    st.session_state.OLMS_DATA_top_oil_mapping = {'Top Oil Temperature': '', 'Ambient Temperature': '', 'HV Current': ''}

if 'Loading_mapping' not in st.session_state:
    st.session_state.Loading_mapping = {'Ambient Temperature': '', 'Ambient Shade Temperature': '', 'HV Current': None}

if 'DGA_mapping' not in st.session_state:
    st.session_state.DGA_mapping = {'H2': "", 'CH4': "", 'C2H2': "", 'C2H6': "", 'C2H4': ""}

if 'asset' not in st.session_state:
    st.session_state.asset = None

if 'Bushings_mapping' not in st.session_state:
    st.session_state.Bushings_mapping = {'Capacitance HV1': '', 'Capacitance HV2': '', 'Capacitance HV3': '',
                                         'Capacitance LV1': '',
                                         'Capacitance LV2': '',
                                         'Capacitance LV3': '', 'tand HV1': '', 'tand HV2': '', 'tand HV3': '',
                                         'tand LV1':  '', 'tand LV2': '', 'tand LV3': ''}




if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False


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



#1st task. Read if there are any assets register for the user



# Set the title of the Streamlit app
if assets.shape[0] > 0:
    tab = st.sidebar.radio("Short Term Asset Management", ["Home", "Import Historical Logs", "Historical Data Dashboard", "Real Time Data Dashboard"])
    st.session_state.assets = assets
else:
    tab = st.sidebar.radio("Navigation", ["Home"])


def home_tab():
    if st.session_state.assets.shape[0]==0:
        st.header("No Registered")
        st.write("Check Documentation for adding new asset on SFTP server.")
    else:
        st.header("Registered Assets")
        st.dataframe(st.session_state.assets.drop(columns=['AssetID','Tool']))
        st.write("Check Documentation for adding new asset on SFTP server.")

if tab=='Home':
    home_tab()
if tab == "Import Historical Logs":
    import_data(assets,measurements_IDs)
    if st.session_state.mapping_confirmed:
        if st.button("Update Machine Learning Models", key="ML Update"):
            if st.session_state.asset is not None:
                for id, asset in assets[assets.AssetName==st.session_state.asset].iterrows():
                    update_anomaly_detection_model(asset,measurements_IDs)
                    update_q50_model(asset,measurements_IDs)
                    update_q90_model(asset,measurements_IDs)

if tab == "Historical Data Dashboard":
    timestamps = pd.date_range(start="2025-01-01", end="2025-01-10", freq="H")
    st.sidebar.header("Select Time Period")

    start_date = st.sidebar.date_input("Start date", timestamps[0].date())
    start_time = st.sidebar.time_input("Start time", time(0, 0))

    end_date = st.sidebar.date_input("End date", timestamps[-1].date())
    end_time = st.sidebar.time_input("End time", time(23, 59))

    # Combine into datetime objects
    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)
    asset =  st.selectbox("Asset:",
                                       assets.AssetName,
                                        key=f"select_asset", index=None  # Unique key for each selectbox
                                    )

    for id, asset_i in assets[assets.AssetName==asset].iterrows():
        Data = read_data_from_DB(asset_i['AssetID'], engine)
        Data = Data[(Data.Timestamp>=start_dt)&(Data.Timestamp<=end_dt)]
        if Data.shape[0]>=1:
            ########################################################################
            ##Calculate DGA Scores##
            DGA_flag = calculate_DGA_flag(Data,measurements_IDs)
            st.write("📉 DGA Scores:")
            st.dataframe(DGA_flag)
            ########################################################################
            ##Calculate Bushing Flags##
            alerted_changes = calculate_bushings_alers(Data,measurements_IDs)
            # Display result
            st.write("🚨 Bushing Capacitance Alerts")
            st.dataframe(alerted_changes.style.format("{:.2f}%"))
            ########################################################################
            ########################################################################
            oil_anomaly_detection_mapping = {'Top Oil Temperature': 'Top Oil Temperature', 'Ambient Temperature': 'Ambient Temperature',
                                      'HV Current': 'HV Current'}
            oil_ids = [meas['MeasurementID'] for id, meas in measurements_IDs.iterrows() if meas['Name'] in oil_anomaly_detection_mapping.keys()]
            Data_oil = Data[Data["MeasurementID"].isin(oil_ids)]
            for meas_id in oil_ids:
                Data_oil.loc[Data_oil.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
            Data_oil = Data_oil.rename(columns={'MeasurementID': 'Measurement'})
            Data_oil.index = pd.DatetimeIndex(Data_oil.Timestamp)
            Data_oil = Data_oil.pivot(index="Timestamp", columns="Measurement", values="Value")
            Data_oil = Data_oil.resample('60min').mean()
            print(Data_oil)

            model = keras.models.load_model('models/top_oil_anomaly/' + asset + '.h5', compile=False)
            with open('models/top_oil_anomaly/'+asset+'_metadata.json', 'r') as f:
                config = json.load(f)
            threshold = config['threshold']
            print("Threshold:", threshold)
            train_ids = Data_oil.index[Data_oil.rolling('60min').count().sum(axis=1) == Data_oil.shape[1]]
            train_ids = train_ids[1:]
            Y = Data_oil.loc[train_ids, 'Top Oil Temperature']
            X = Data_oil.shift(1).loc[train_ids]
            X= X.dropna(axis=0)
            Y = Y.loc[X.index]
            alarms = predict_top_oil(X,Y,model, threshold)
            st.write("🚨Top Oil Temperature anomalies detected")
            st.dataframe(alarms[alarms['alarms']==True])
        else:
            st.write(f"🚨No data available for asset:{asset_i['AssetName']}")

if tab == "Real Time Data Dashboard":
    st.header('Real Time Data Dashboard')
    asset =  st.selectbox("Asset:",
                                       assets.AssetName,
                                        key=f"select_asset", index=None  # Unique key for each selectbox
                                    )
    for id, asset_i in assets[assets.AssetName==asset].iterrows():
        Data = read_data_from_DB(asset_i['AssetID'], engine)
        latest_ts = Data.Timestamp.max()
        Data = Data[(Data.Timestamp >= latest_ts - pd.Timedelta(hours=48)) & (Data.Timestamp <= latest_ts)]
        st.write(f"Real Time Dashboard: Latest data received at: {latest_ts}")
        DGA_flag = calculate_DGA_flag(Data,measurements_IDs)

        Bushings_mapping = {'Capacitance HV1': 'Capacitance HV1', 'Capacitance HV2': 'Capacitance HV2',
                                        'Capacitance HV3': 'Capacitance HV3','Capacitance LV1': 'Capacitance LV1',
                                        'Capacitance LV2': 'Capacitance LV2', 'Capacitance LV3': 'Capacitance LV3',
                                        'tand HV1': 'tand HV1', 'tand HV2': 'tand HV2', 'tand HV3': 'tand HV3',
                                                     'tand LV1':  'tand LV1', 'tand LV2': 'tand LV2', 'tand LV3': 'tand LV3'}
        Bushings_ids = [meas['MeasurementID'] for id, meas in measurements_IDs.iterrows()
                                if meas['Name'] in Bushings_mapping.keys()]
        Data_Bushings = Data[Data["MeasurementID"].isin(Bushings_ids)]
        for meas_id in Bushings_ids:
            Data_Bushings.loc[Data_Bushings.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
        Data_Bushings = Data_Bushings.rename(columns={'MeasurementID': 'Measurement'})
        if Data_Bushings.shape[0]>=48:
            Alarms_Bushings = calculate_bushings_alers(Data,measurements_IDs)
        else:
            Data_Bushings = pd.DataFrame()
        ########################################################################
        ########################################################################
        alarms_oil, threshold, Data_oil = calculate_top_oil_alarms(Data,measurements_IDs)

    tabs_real_time = st.tabs(['Alarms','Top Oil Forecast'])
    with tabs_real_time[0]:
        col1, col2, col3 = st.columns([20, 20, 20])
        with col1:
            st.subheader("Dissolved Gas Analysis")
            if DGA_flag.shape[0]>=1:
                st.markdown(display_light(not ((DGA_flag['SCORE'] <= 3).any())), unsafe_allow_html=True)
                if (DGA_flag['SCORE'] <= 3).any():
                    DGA_flag['SCORE'] = 'ok'
                    st.write("Status ok. No action suggested based on DGA data of last 48 hours")
                elif (3 < DGA_flag['SCORE']).any() & (DGA_flag['SCORE'] <= 5).any():
                    DGA_flag['SCORE'] = 'minor warning'
                    st.write("Gas levels considerable but within limits. Increase monitoring")
                elif (5 < DGA_flag['SCORE']).any() & (DGA_flag['SCORE'] <= 7).any():
                    DGA_flag['SCORE'] = 'warning'
                    st.write("Gas levels considerable. Maintenance actions suggested")
                else:
                    DGA_flag['SCORE'] = 'critical'
                    st.write("Gas levels critical. Maintenance actions suggested")
                st.write("DGA Results")
                st.write(DGA_flag)
            else:
                st.write("Insufficient Data for Dissolved Gas Analysis")
        with col2:
            st.subheader("Bushings Analysis")
            print('Here')
            if not(Data_Bushings.empty):
                st.markdown(display_light(not(Alarms_Bushings.empty)), unsafe_allow_html=True)
                if (Alarms_Bushings.empty):
                    st.write("No warnings on the Bushings on past 48h. Bushing status ok. No action suggested based on DGA data of last 48 hours")
                else:
                    st.write("Warnings on the Bushings")
                    st.dataframe(Alarms_Bushings.style.format("{:.2f}%"))
            else:
                st.write("Insufficient Data for Bushing condition assessment")
        with col3:
            st.subheader("Oil Anomaly Detection")
            if alarms_oil.shape[0]>=1:
                st.markdown(display_light(not(alarms_oil[alarms_oil.alarms==True].empty)), unsafe_allow_html=True)
                if (alarms_oil[alarms_oil.alarms==True].empty):
                    st.write("No anomaly detected in top oil temperature.")
                else:
                    st.write("Anomalies detected in Top Oil Temperature")
                    st.dataframe(alarms_oil[alarms_oil.alarms==True])
                st.write('Model Prediction latest 48 hours')
                st.components.v1.html(html_error_plot(alarms_oil['Deviation (Celsius)'], threshold), height=500)
            else:
                st.write("Insufficient data for top-oil anomaly detection")
    with tabs_real_time[1]:
        st.subheader("Top Oil Temperature Forecast")
        if Data_oil.shape[0]>=24:
            model_90 = keras.models.load_model('models/oil_temp_forecast/' + asset + '_90.keras', compile=False)
            model_50 = keras.models.load_model('models/oil_temp_forecast/' + asset + '_50.keras', compile=False)
            predictions_90 = predict_q90_model(model_90,Data_oil)
            predictions_50 = predict_q90_model(model_50,Data_oil)
            predictions = pd.DataFrame(np.hstack((predictions_50.transpose(),predictions_90.transpose())),
                                       columns=['Q50','Q90'],index=[Data_oil.index.max()+timedelta(hours=i)
                                                                                                             for i in range(1,7)])
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            threshold = st.slider('Select Top Oil Temperature Limit', min_value=50.0, max_value=90.0, value=60.0, step=1.0)
            st.components.v1.html(html_future_oil_temp_plot(predictions,threshold), height=600, width=1200, scrolling=True)
            print(predictions)
            st.write("Probability of Failure")
            st.dataframe(probability_to_exceed(threshold, predictions['Q50'], (predictions['Q90'] - predictions['Q50']) / 1.2816))
        else:
            st.write("Insufficient data for Top-Oil Temperature Forecast")










