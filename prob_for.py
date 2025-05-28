import pandas as pd
import math
from pathlib import Path
from AAM_predict_toolbox import predict_oil_future, html_future_oil_temp_plot, train_models_current, \
    compute_warning_on_bushing, compute_warning_on_DGA, display_light, generate_training_data_oil, \
    prepare_model_top_oil, predict_top_oil, html_error_plot
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
import numpy as np
import os
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error
import tensorflow.keras.backend as K
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam





user = "'IPTO'"
#Create Engine
server = '147.102.30.47'            # or IP address / named instance
database = 'opentunity_dev'
username = 'opentunity'
password = '0pentunity44$$'
conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(conn_str)


def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss


def read_data_from_DB(asset_ID, engine):
    data = pd.read_sql_query( f"""
                SELECT * FROM Measurements
                WHERE AssetID = {asset_ID}
                """,
                con=engine)
    return data

def get_assets_of_user(user):
    query = f"SELECT * FROM assets WHERE Owner = {user} and Tool = 'ST_AAM'"

    # Read the result into a DataFrame
    user_assets = pd.read_sql(query, con=engine)
    return user_assets

def get_measurements_IDs(engine):

    measurement_IDS = pd.read_sql("SELECT MeasurementID,Name FROM MeasurementTypes", con=engine)
    return measurement_IDS

def get_asset_ID(engine,asset_name):
    asset_ID = pd.read_sql("SELECT AssetID FROM Assets where AssetName="+"'"+asset_name+"'", con=engine)
    return asset_ID.values[0][0]


def create_multistep_sequences(X, y, input_len=24, output_len=6):
    Xs, ys = [], []
    for i in range(len(X) - input_len - output_len):
        Xs.append(X[i:i+input_len])
        ys.append(y[i+input_len:i+input_len+output_len].flatten())
    return np.array(Xs), np.array(ys)


def build_model(hp):
    model = Sequential()

    # Tune number of LSTM units
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=128, step=32),
                   input_shape=(X_seq.shape[1], X_seq.shape[2])))

    # Optional: add another Dense or dropout
    model.add(Dense(hp.Int('dense_units', 16, 64, step=16), activation='relu'))

    # Output layer
    model.add(Dense(y_seq.shape[1]))  # Multi-step output

    # Tune learning rate
    lr = hp.Choice('learning_rate', [1e-4, 1e-3, 1e-2])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model

def train_mean_model(Data):
    df = Data.pivot(columns='Measurement', values='Value')

    # # Select features and target
    features = ['HV Current', 'Ambient Temperature']  # adjust as needed
    target = 'Top Oil Temperature'
    df = df[features+[target]]
    df = df.resample('60min').mean()
    df.dropna(inplace=True)
    # df = pd.DataFrame()
    #
    # # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    #
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target]])
    sequence_length = 24  # e.g., use past 24 hours
    X_seq, y_seq = create_multistep_sequences(X_scaled, y_scaled, input_len=24, output_len=6)
    #
    print('a')
    #
    model = Sequential([
        LSTM(128, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dense(32),  # 6 output values, one per hour
        Dense(6)  # 6 output values, one per hour
    ])

    model.compile(optimizer='adam', loss='mse')
    #
    model.summary()
    #
    model.fit(X_seq, y_seq, epochs=40, batch_size=32, validation_split=0.2)
    y_pred_scaled = model.predict(X_seq)  # Predict for latest input
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq)

    return model, y_pred, y_true


def train_q90_model(Data):
    df = Data.pivot(columns='Measurement', values='Value')

    # # Select features and target
    features = ['HV Current', 'Ambient Temperature']  # adjust as needed
    target = 'Top Oil Temperature'
    df = df[features+[target]]
    df = df.resample('60min').mean()
    df.dropna(inplace=True)
    # df = pd.DataFrame()
    #
    # # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    #
    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target]])
    X_seq, y_seq = create_multistep_sequences(X_scaled, y_scaled, input_len=24, output_len=6)
    #
    print('a')
    #
    model = Sequential([
        LSTM(128, input_shape=(X_seq.shape[1], X_seq.shape[2])),
        Dense(32),  # 6 output values, one per hour
        Dense(6)  # 6 output values, one per hour
    ])
    model.compile(optimizer='adam', loss=quantile_loss(0.9))
    #
    model.summary()
    #
    model.fit(X_seq, y_seq, epochs=40, batch_size=32, validation_split=0.2)

    y_pred_scaled = model.predict(X_seq)  # Predict for latest input
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_seq)

    return model, y_pred, y_true

assets = get_assets_of_user(user)
measurements_IDs = get_measurements_IDs(engine)

asset = assets.loc[0]

Data = read_data_from_DB(asset['AssetID'], engine)
for meas_id in Data.MeasurementID.unique().tolist():
    Data.loc[Data.MeasurementID==meas_id,'MeasurementID']=measurements_IDs.loc[measurements_IDs.MeasurementID==meas_id,'Name'].values[0]
Data.index = pd.DatetimeIndex(Data.Timestamp)
Data.drop(columns=['AssetID','Timestamp'],inplace=True)
Data = Data.rename(columns={'MeasurementID': 'Measurement'})
df = Data.pivot(columns='Measurement', values='Value')

print('a')


model_90,y_pred_90, y_true_90 = train_q90_model(Data)
model,y_pred, y_true = train_mean_model(Data)



horizons = 6  # 6 forecast steps
import matplotlib.pyplot as plt


for i in range(horizons):
    plt.figure(figsize=(6, 3))
    plt.plot(y_pred[:, i], label='50', color='blue')
    plt.plot(y_pred_90[:, i], label='90', color='blue')
    plt.plot(y_true[:, i], label='actual', color='red')
    plt.title(f'Forecast {i+1} hour(s) ahead')
    plt.legend()
    plt.tight_layout()
    plt.show()

for i in range(horizons):
    mae = mean_absolute_percentage_error(y_true[:, i], y_pred[:, i])
    mse = mean_squared_error(y_true[:, i], y_pred[:, i])
    r2 = r2_score(y_true[:, i], y_pred[:, i])
    print(f"⏱️ Forecast {i+1} hour(s) ahead:")
    print(f"   MAPE: {mae:.4f}")
    print(f"   MSE: {mse:.4f}")
    print(f"   R² : {r2:.4f}")
