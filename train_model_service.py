from sqlalchemy import create_engine
import os
import json
from datetime import datetime, timedelta, time
import pandas as pd
from AAM_predict_toolbox import train_q90_model,  generate_training_data_oil, prepare_model_top_oil, train_mean_model

#######To be replaced by a loop for all users########
user = "'IPTO'"
#Create Engine
server = '147.102.30.47'            # or IP address / named instance
database = 'opentunity_dev'
username = 'opentunity'
password = '0pentunity44$$'
conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(conn_str)
#############################################################
def get_assets_of_user(user):
    query = f"SELECT * FROM assets WHERE Owner = {user} and Tool = 'ST_AAM'"

    # Read the result into a DataFrame
    user_assets = pd.read_sql(query, con=engine)
    return user_assets

def is_model_stale(metadata_path, days=7):
    #check if model is trained over the last  X  days, default = 7

    if not os.path.isfile(metadata_path):
        return True  # No model or metadata found → needs retraining

    with open(metadata_path) as f:
        metadata = json.load(f)

    trained_on = datetime.strptime(metadata["trained_on"], "%Y-%m-%d_%H-%M-%S")
    return datetime.now() - trained_on > timedelta(days=days)

def read_data_from_DB(asset_ID, engine):
    data = pd.read_sql_query( f"""
                SELECT * FROM Measurements
                WHERE AssetID = {asset_ID}
                """,
                con=engine)
    return data

def get_asset_ID(engine,asset_name):
    asset_ID = pd.read_sql("SELECT AssetID FROM Assets where AssetName="+"'"+asset_name+"'", con=engine)
    return asset_ID.values[0][0]

def train_top_oil_anomaly_detection_model(Data,top_oil_mapping,DGA_mapping,Bushings_mapping):
    X, Y = generate_training_data_oil(Data,top_oil_mapping,DGA_mapping, Bushings_mapping)
    model_oil, oil_threshold = prepare_model_top_oil(X, Y)
    #Store model and metadata
    return model_oil, oil_threshold


def train_anomaly_detection_model(asset,measurements_IDs):
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

def get_measurements_IDs(engine):

    measurement_IDS = pd.read_sql("SELECT MeasurementID,Name FROM MeasurementTypes", con=engine)
    return measurement_IDS

def train_q90(asset,measurements_IDs):
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


def train_q50(asset,measurements_IDs):
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

def retrain_stale_models(assets,measurements_IDs):
    ##Check and re-train the models of every asset in assets
    for id, asset in assets.iterrows():
        if is_model_stale('models/top_oil_anomaly/'+asset.AssetName+'_metadata.json',days=7):
            train_anomaly_detection_model(asset,measurements_IDs)
        if is_model_stale('models/oil_temp_forecast/'+asset.AssetName+'_90_metadata.json',days=7):
            train_q90(asset,measurements_IDs)
        if is_model_stale('models/oil_temp_forecast/'+asset.AssetName+'_50_metadata.json',days=7):
            train_q50(asset,measurements_IDs)

if __name__ == '__main__':
    assets = get_assets_of_user(user)
    measurements_IDs = get_measurements_IDs(engine)
    retrain_stale_models(assets,measurements_IDs)
