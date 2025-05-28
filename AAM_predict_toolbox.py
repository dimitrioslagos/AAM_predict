import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras
from tf_keras import layers
import plotly.express as px
import os
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_percentage_error
import tensorflow.keras.backend as K
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats

maxV = {'Top Oil Temperature': 70, 'Ambient Temperature': 50, 'Ambient Shade Temperature': 50, 'HV Current': 300}
maxV = pd.Series(maxV)
threshold = {'Top Oil Temperature': 60}



def quantile_90_bins(OIL, current_bins, temperature_bins):
    filtered_data = pd.DataFrame()

    for i in range(len(current_bins) - 1):
        for j in range(len(temperature_bins) - 1):
            bin_data = OIL[
                (OIL['HV Current'] >= current_bins[i]) &
                (OIL['HV Current'] < current_bins[i + 1]) &
                (OIL['Ambient Temperature'] >= temperature_bins[j]) &
                (OIL['Ambient Temperature'] < temperature_bins[j + 1])
                ]

            if not bin_data.empty:  # Check if DataFrame is not empty (contains data)
                Temp_quantile_005 = bin_data['Top Oil Temperature'].quantile(0.05)
                Temp_quantile_095 = bin_data['Top Oil Temperature'].quantile(0.95)
                bin_data = bin_data[
                    (bin_data['Top Oil Temperature'] >= Temp_quantile_005) &
                    (bin_data['Top Oil Temperature'] <= Temp_quantile_095)
                    ]
                filtered_data = pd.concat([filtered_data, bin_data])
    filtered_data = filtered_data.sort_index()

    return filtered_data

def remove_extremes_OIL(DATA, OLMS_DATA_top_oil_mapping):
    Data_oil = DATA[DATA["Measurement"].isin(OLMS_DATA_top_oil_mapping.keys())]
    Data_oil['Timestamp'] = Data_oil.index
    OIL = Data_oil.pivot(index="Timestamp", columns="Measurement", values="Value")

    maxT= 10 * np.ceil(OIL['Ambient Temperature'].max()/ 10)
    minT= 10 * np.floor(OIL['Ambient Temperature'].min()/ 10)
    maxI= 5 * np.ceil(OIL['HV Current'].max()/ 5)
    minI= 5 * np.floor(OIL['HV Current'].min()/ 5)
    current_bins = np.arange(minI, maxI + 5, 5)
    temperature_bins = np.arange(minT, maxT + 10, 10)
    OIL_filtered = quantile_90_bins(OIL, current_bins, temperature_bins)
    OIL_filtered.dropna(inplace=True)
    return OIL_filtered

def probability_to_exceed(value, mean, std):
    # Calculate the z-score
    print(value, mean, std)
    z = (value - mean) / std
    print(z)
    # Get the cumulative probability up to the value
    cdf_value = stats.norm.cdf(z.values.tolist())
    print(cdf_value)
    # The probability to exceed the value is 1 - CDF
    exceed_probability = cdf_value

    return pd.DataFrame(exceed_probability.reshape(1, 6), index=['Failure Probability (%)'], columns=z.index)

def html_future_oil_temp_plot(OIL_temp):
    # Create Plotly figure
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=OIL_temp.index, y=OIL_temp['max'].values, mode='lines', name='max'))
    fig.add_trace(go.Scatter(x=OIL_temp.index, y=OIL_temp['mean'].values, mode='lines', name='mean'))
    fig.add_trace(go.Scatter(x=OIL_temp.index, y=OIL_temp['min'].values, mode='lines', name='min'))

    # Fill the area between min and max lines
    fig.add_trace(go.Scatter(
        x=OIL_temp.index,
        y=OIL_temp['max'].values,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0)'),  # Transparent line for the max line
        showlegend=False,
        name='max'
    ))

    fig.add_trace(go.Scatter(
        x=OIL_temp.index,
        y=OIL_temp['min'].values,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0)'),  # Transparent line for the min line
        fill='tonexty',  # Fill area between this line and the previous line
        fillcolor='rgba(128, 128, 128, 0.5)',  # Gray color with some transparency
        showlegend=False,
        name='min'
    ))

    # Add the second line
    fig.add_trace(go.Scatter(x=OIL_temp.index,
                             y=threshold['Top Oil Temperature'] * OIL_temp['min'].values / OIL_temp['min'].values,
                             mode='lines', name='Treshold'))

    # Customize the layout (optional)
    fig.update_layout(title="Oil Temperature Forecast (°C)",
                      showlegend=False,
                      template="plotly_white")
    #fig.show()
    return fig.to_html()

def html_error_plot(error, threshold):
    # Create Plotly figure
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=error.index, y=error.values, mode='lines', name='Error (C)'))

    # Add the second line
    fig.add_trace(go.Scatter(x=error.index, y=threshold * error.values / error.values, mode='lines', name='Treshold'))

    # Add the third line 
    fig.add_trace(go.Scatter(x=error.index, y=-threshold * error.values / error.values, mode='lines', name='Treshold'))

    # Customize the layout (optional)
    fig.update_layout(title="Oil Temperature Model Error (°C)",
                      showlegend=False,
                      template="plotly_white")

    return fig.to_html()

def display_light(value):
    if value:  # True -> Yellow light
        return '<div style="width: 50px; height: 50px; background-color: yellow; border-radius: 50%;"></div>'
    else:  # False -> Green light
        return '<div style="width: 50px; height: 50px; background-color: green; border-radius: 50%;"></div>'

def build_regression_model(input_shape):
    model = tf_keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    return model

def build_regression_model_I(input_shape):
    model = tf_keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    return model

def loss_mse(y_true, y_pred):
    e = y_true - y_pred
    return tf.reduce_mean(e * e)

def quantile_loss(q):
    """Creates a quantile loss function for a given quantile q.

    Args:
        q (float): The quantile to predict (e.g., 0.5 for median).

    Returns:
        loss function: A quantile loss function for use with model compilation.
    """

    def loss(y_true, y_pred):
        error = (y_true - y_pred)
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

    return loss

def train_model_top_oil(X_train, y_train):
    print(X_train)
    maxX = maxV[X_train.columns]
    # Instantiate the model
    input_shape = (X_train.shape[1],)  # Assuming X_train is your feature matrix
    model = build_regression_model(input_shape)
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Train the model
    history = model.fit(X_train / maxX, y_train / maxX['Top Oil Temperature'], epochs=50, verbose=0)
    return model

def create_multistep_sequences(X, y, input_len=24, output_len=6):
    Xs, ys = [], []
    for i in range(len(X) - input_len - output_len):
        Xs.append(X[i:i+input_len].values)
        ys.append(y[i+input_len:i+input_len+output_len].values.flatten())
    return np.array(Xs), np.array(ys)

def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)
    return loss


def train_q90_model(Data):
    df = Data.pivot(columns='Measurement', values='Value')

    # # Select features and target
    features = ['HV Current', 'Ambient Temperature']  # adjust as needed
    target = 'Top Oil Temperature'
    df = df[features+[target]]
    df = df.resample('60min').mean()
    df.dropna(inplace=True)

    # # Normalize features
    X_scaled = df[features]/maxV[features]
    y_scaled = df[target]/maxV[target]
    X_seq, y_seq = create_multistep_sequences(X_scaled, y_scaled, input_len=24, output_len=6)

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


    return model


def train_mean_model(Data):
    df = Data.pivot(columns='Measurement', values='Value')

    # # Select features and target
    features = ['HV Current', 'Ambient Temperature']  # adjust as needed
    target = 'Top Oil Temperature'
    df = df[features+[target]]
    df = df.resample('60min').mean()
    df.dropna(inplace=True)

    #
    X_scaled = df[features]/maxV[features]
    y_scaled = df[target]/maxV[target]
    sequence_length = 24  # e.g., use past 24 hours
    X_seq, y_seq = create_multistep_sequences(X_scaled, y_scaled, input_len=24, output_len=6)

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

    return model






def prepare_model_top_oil(X, Y):
    # Specify the directory path
    model = train_model_top_oil(X, Y)
    maxX = maxV[X.columns]
    ypred = model.predict(X / maxX.values) * maxX['Top Oil Temperature']
    mean_error = (Y - ypred.reshape(ypred.shape[0])).mean()
    std_error = (Y - ypred.reshape(ypred.shape[0])).std()
    threshold = mean_error + 3 * std_error
    return model, threshold


def predict_top_oil(X_test, y_test, model, threshold):
    # model = tf_keras.models.load_model('Models/Top_Oil',custom_objects={'loss_mse':  loss_mse})
    maxX = maxV[X_test.columns]
    ypred = model.predict(X_test / maxX.values) * maxX['Top Oil Temperature']
    DATA = pd.DataFrame(columns=['Real', 'Estimate'])
    DATA['Real'] = y_test
    DATA['Estimate'] = ypred
    issues, errors = anomaly_detection_in_oil_temp(ypred, y_test, threshold)
    return issues, errors


def predict_T_bushing(model, X_test, y_test):
    maxX = pd.Series([50, 50, 50, 288, 50, 50, 288], index=X_test.columns)
    ypred = model.predict(X_test / maxX.values) * maxX.values[0]
    print(loss_mse(y_test, ypred.reshape(ypred.shape[0])))
    DATA = pd.DataFrame(columns=['Real', 'Estimate'])
    DATA['Real'] = y_test
    DATA['Estimate'] = ypred
    # # Create a scatter plot
    fig = px.line(DATA, x=DATA.index, y='Estimate', title="Sample Scatter Plot")
    fig.add_scatter(x=DATA.index, y=DATA['Real'], mode='lines', name='Real', line=dict(color='red'))

    # Save the plot as an HTML file
    fig.write_html("scatter_plot.html")

    # Show the plot (optional, will open the plot in a browser)
    fig.show()
    return 0

def compute_hydrogen_condition_state(h2_value):
    if -0.01 <= h2_value <= 20.00:
        return 0
    elif 20.00 < h2_value <= 40.00:
        return 2
    elif 40.00 < h2_value <= 100.00:
        return 4
    elif 100.00 < h2_value <= 200.00:
        return 10
    elif 200.00 < h2_value <= 10000.00:
        return 16
    else:
        return 0  # Handle cases outside of the specified ranges, if needed

def compute_methane_condition_state(ch4_value):
    if -0.01 <= ch4_value <= 10.00:
        return 0
    elif 10.00 < ch4_value <= 20.00:
        return 2
    elif 20.00 < ch4_value <= 50.00:
        return 4
    elif 50.00 < ch4_value <= 150.00:
        return 10
    elif 150.00 < ch4_value <= 10000.00:
        return 16
    else:
        return None  # Handle cases outside the specified ranges, if needed

def compute_ethylene_condition_state(c2h4_value):
    if -0.01 <= c2h4_value <= 10.00:
        return 0
    elif 10.00 < c2h4_value <= 20.00:
        return 2
    elif 20.00 < c2h4_value <= 50.00:
        return 4
    elif 50.00 < c2h4_value <= 150.00:
        return 10
    elif 150.00 < c2h4_value <= 10000.00:
        return 16
    else:
        return 0  # Handle cases outside the specified ranges, if needed

def compute_ethane_condition_state(c2h6_value):
    if -0.01 <= c2h6_value <= 10.00:
        return 0
    elif 10.00 < c2h6_value <= 20.00:
        return 2
    elif 20.00 < c2h6_value <= 50.00:
        return 4
    elif 50.00 < c2h6_value <= 150.00:
        return 10
    elif 150.00 < c2h6_value <= 10000.00:
        return 16
    else:
        return 0  # Handle cases outside the specified ranges, if needed

def compute_acetylene_condition_state(c2h2_value):
    if -0.01 <= c2h2_value <= 1.00:
        return 0
    elif 1.00 < c2h2_value <= 5.00:
        return 2
    elif 5.00 < c2h2_value <= 20.00:
        return 4
    elif 20.00 < c2h2_value <= 100.00:
        return 8
    elif 100.00 < c2h2_value <= 10000.00:
        return 10
    else:
        return 0

def compute_normal_scenarios(DGA):
    DGA['C2H2_State'] = DGA['C2H2'].apply(compute_acetylene_condition_state)
    DGA['C2H6_State'] = DGA['C2H6'].apply(compute_ethane_condition_state)
    DGA['C2H4_State'] = DGA['C2H4'].apply(compute_ethylene_condition_state)
    DGA['CH4_State'] = DGA['CH4'].apply(compute_methane_condition_state)
    DGA['H2_State'] = DGA['H2'].apply(compute_hydrogen_condition_state)
    SCORE = 50 * DGA['H2_State'] + 30 * (DGA['CH4_State'] + DGA['C2H4_State'] + DGA['C2H6_State']) + 120 * DGA[
        'C2H2_State']
    ids = (SCORE / 120) <= 3
    return ids

def prepare_DGA_df(DATA, OLMS_mapping):
    DGA = pd.DataFrame(columns=['H2', 'CH4', 'C2H2', 'C2H6', 'C2H4'])
    for col in DGA.columns:
        DGA[col] = DATA.loc[DATA.Measurement == OLMS_mapping[col], 'Value'].resample('30min').mean()

    return DGA

def prepare_bushing_capacitance_df(DATA, bushing_mapping):
    Capacitances = pd.DataFrame(columns=['Capacitance HV1', 'Capacitance HV2', 'Capacitance HV3',
                                         'Capacitance LV1', 'Capacitance LV2','Capacitance LV3'])
    for col in Capacitances.columns:
        Capacitances[col] = DATA.loc[DATA.Measurement == bushing_mapping[col], 'Value'].resample('180min').mean()

    return Capacitances


def prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping):
    OIL = pd.DataFrame(columns=['Top Oil Temperature', 'Ambient Temperature', 'HV Current'])
    for col in OIL.columns:
        OIL[col] = DATA.loc[DATA.Measurement == OLMS_DATA_top_oil_mapping[col], 'Value'].resample('30min').mean()
    return OIL


def data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping, bushings_mapping):
    OIL = remove_extremes_OIL(DATA, OLMS_DATA_top_oil_mapping)
    OIL = OIL.resample('30min').mean()
    OIL.dropna(inplace=True)
    DGA = prepare_DGA_df(DATA, DGA_mapping).dropna()
    if DGA.shape[0]>=1:
        ids = compute_normal_scenarios(DGA)
        OIL = OIL.loc[DGA.index[ids.values].intersection(OIL.index)]

    Capacitances = prepare_bushing_capacitance_df(DATA,bushing_mapping=bushings_mapping)
    if Capacitances.shape[0]>=1:
        Capacitances_perc_change = Capacitances.pct_change()
        mask = (Capacitances_perc_change <= -0.10) | (Capacitances_perc_change >= 0.05)
        issue_times = Capacitances.index[mask.any(axis=1)]
        time_window = pd.Timedelta(hours=3)
        mask2 = pd.Series(False, index=OIL.index)
        for time in issue_times:
            mask2 |= (OIL.index >= time - time_window) & (OIL.index <= time + time_window)
        OIL = OIL.drop(index = OIL.index[mask2])
    OIL.dropna(inplace=True)
    return OIL


def generate_training_data_oil(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping,bushing_mapping):
    OIL = data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping,bushing_mapping)
    OIL = OIL.resample('60min').mean()

    train_ids = OIL.index[OIL.rolling('60min').count().sum(axis=1) == OIL.shape[1]]

    train_ids = train_ids[1:]
    Y = OIL.loc[train_ids, 'Top Oil Temperature']
    X = OIL.shift(1).loc[train_ids]
    X= X.dropna(axis=0)
    Y = Y.loc[X.index]

    return X, Y


def prepare_current_relevant_data(DATA, Loading_mapping):
    Loading = pd.DataFrame(columns=['Ambient Temperature', 'Ambient Shade Temperature', 'HV Current'])
    # Loading_mapping = {'Ambient Temperature': 'Ampient Sun',
    #                    'Ambient Shade Temperature': 'Ampient Shade',
    #                    'HV Current': 'HV Load Current'}
    for col in Loading.columns:
        Loading[col] = DATA.loc[DATA.Measurement == Loading_mapping[col], 'Value'].resample('1h').mean()
    return Loading


def generate_current_training_data(DATA, Loading_mapping, horizon):
    Loading = prepare_current_relevant_data(DATA, Loading_mapping)
    flag = (Loading.rolling('26h').count().sum(axis=1) == (26) * Loading.shape[1]) & \
           (Loading.index.isin(Loading.index.shift(-horizon, freq='h')))
    X = [Loading.shift(1).loc[flag, 'HV Current']]
    for i in range(2, 6, 1):
        X.append(Loading.shift(i).loc[flag, 'HV Current'].rename('HV Current_' + str(i)))
    X = pd.concat(X, axis=1)
    X['h'] = (Loading.index[flag] + pd.Timedelta(hours=horizon - 1)).hour
    X['m'] = (Loading.index[flag] + pd.Timedelta(hours=horizon - 1)).month
    X['d'] = (Loading.index[flag] + pd.Timedelta(hours=horizon - 1)).dayofweek
    # X['Ambient Temperature'] = Loading.loc[flag,'Ambient Temperature']#Loading.shift(1).loc[flag,'Ambient Temperature']
    Y = Loading.loc[Loading.index[flag] + pd.Timedelta(hours=horizon - 1), 'HV Current']
    Y.index = X.index
    # Yest = pd.Series(index=Y.index,name=Y.name)
    # Loading['month'] = Loading.index.month
    # Loading['hour'] = Loading.index.hour
    # Loading['day'] = Loading.index.dayofweek
    # Data_Aggr_std =Loading.groupby(['month', 'hour','day'])['HV Current'].std()
    # Data_Aggr_m = Loading.groupby(['month', 'hour', 'day'])['HV Current'].mean()
    # for i in Y.index:
    # Yest.loc[i] = Data_Aggr_m[i.month,i.hour,i.dayofweek]+3*Data_Aggr_std[i.month,i.hour,i.dayofweek]
    return X, Y


def anomaly_detection_in_oil_temp(y_pred, y_true, threshold):
    MRi = y_true - pd.Series(y_pred.reshape(y_pred.shape[0]), index=y_true.index)
    MR = MRi.diff().dropna()
    Flags = (MR >= threshold).rolling(window=4).mean() >= 0.75

    return Flags, MR


def compute_warning_on_bushing(t, DATA, Bushings_mapping):
    Bushings = pd.DataFrame(columns=['Cap H1', 'Cap H2', 'Cap H3', 'Cap Y1', 'Cap Y2', 'Cap Y3',
                                     'tand H1', 'tand H2', 'tand H3', 'tand Y1', 'tand Y2', 'tand Y3'])
    # Bushings_mapping = {'Cap H1': 'BUSHING H1 Capacitance',
    #                     'Cap H2': 'BUSHING H2 Capacitance',
    #                     'Cap H3': 'BUSHING H3 Capacitance',
    #                     'Cap Y1': 'BUSHING Y1 Capacitance',
    #                     'Cap Y2': 'BUSHING Y2 Capacitance',
    #                     'Cap Y3': 'BUSHING Y3 Capacitance',
    #                     'tand H1': 'BUSHING H1 Tan delta',
    #                     'tand H2': 'BUSHING H2 Tan delta',
    #                     'tand H3': 'BUSHING H3 Tan delta',
    #                     'tand Y1': 'BUSHING Y1 Tan delta',
    #                     'tand Y2': 'BUSHING Y2 Tan delta',
    #                     'tand Y3': 'BUSHING Y3 Tan delta'}
    for col in Bushings.columns:
        Bushings[col] = DATA.loc[DATA.Measurement == Bushings_mapping[col], 'Value'].resample('30min').mean()
    print(Bushings)

    Bushings_last = Bushings[(Bushings.index >= (t - pd.Timedelta(days=7))) & (Bushings.index <= t)]
    value = Bushings_last[-6:].mean()
    mean = Bushings_last[:-6].mean()
    abs_change = 100 * (mean - value).abs() / mean
    warning = abs_change >= 10
    Message = pd.DataFrame(columns=warning.index)
    Message.loc[0, Message.columns[warning]] = 'Warning'
    Message.loc[0, Message.columns[warning == False]] = 'OK'
    Message.index = ['Status']
    return Message


def compute_warning_on_DGA(DATA):
    DGA_mapping = {'H2': "H2", 'CH4': "CH4", 'C2H2': "C2H2", 'C2H6': "C2H6", 'C2H4': "C2H4"}
    DGA = prepare_DGA_df(DATA, DGA_mapping)
    DGA.fillna(0,inplace=True)
    DGA['C2H2_Score'] = DGA['C2H2'].apply(compute_acetylene_condition_state)
    DGA['C2H6_Score'] = DGA['C2H6'].apply(compute_ethane_condition_state)
    DGA['C2H4_Score'] = DGA['C2H4'].apply(compute_ethylene_condition_state)
    DGA['CH4_Score'] = DGA['CH4'].apply(compute_methane_condition_state)
    DGA['H2_Score'] = DGA['H2'].apply(compute_hydrogen_condition_state)
    DGA['SCORE'] = 50 * DGA['H2_Score'] + 30 * (
                DGA['CH4_Score'] + DGA['C2H4_Score'] + DGA['C2H6_Score']) + 120 * DGA['C2H2_Score']
    return DGA
