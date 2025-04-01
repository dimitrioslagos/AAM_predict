import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras
from tf_keras import layers
import plotly.express as px
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

maxV = {'Top Oil Temperature': 70, 'Ambient Temperature': 50, 'Ambient Shade Temperature': 50, 'HV Current': 300}
maxV = pd.Series(maxV)

threshold = {'Top Oil Temperature': 60}

import scipy.stats as stats


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


###
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


# Function to generate error plot
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


# Function to display light based on boolean value
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
    model.compile(optimizer='adam', loss=loss_mse)
    # Train the model
    history = model.fit(X_train / maxX, y_train / maxX['Top Oil Temperature'], epochs=50, verbose=0)
    # model.save('Models/Top_Oil')
    return model


def predict_quantiles(model, X, quantiles):
    predictions = []
    for q in quantiles:
        pred = np.percentile([tree.predict(X) for tree in model.estimators_], q * 100, axis=0)
        predictions.append(pred)
    return np.array(predictions).T


def train_model_current(X_train, y_train):
    maxX = pd.Series(index=X_train.columns)
    for i in maxX.index:
        if 'HV Current' in i:
            maxX[i] = maxV['HV Current']
        elif i == 'h':
            maxX[i] = 23
        elif i == 'm':
            maxX[i] = 12
        elif i == 'd':
            maxX[i] = 6
        else:
            maxX[i] = maxV[i]
    # Instantiate the model
    if y_train.isna().any():
        X_train.drop(index=y_train.index[y_train.isna()], inplace=True)
        y_train.drop(index=y_train.index[y_train.isna()], inplace=True)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model


def predict_I_value(model, X_test):
    quantiles = [0.5, 0.75, 0.9, 0.99]
    quantile_predictions = predict_quantiles(model, X_test, quantiles)
    quantile_df = pd.DataFrame(quantile_predictions, columns=[f'quantile_{q}' for q in quantiles])
    return quantile_df


def train_models_current(DATA, Loading_mapping, horizon):
    models = []
    for i in range(1, horizon + 1):
        X, Y = generate_current_training_data(DATA, Loading_mapping, horizon=i)
        models.append(train_model_current(X, Y))
    return models


def predict_Currents(models, DATA, Loading_mapping, horizon, t):
    ##
    quantiles = [0.5]
    ##    ##
    qs = pd.DataFrame(index=[t + pd.Timedelta(hours=i) for i in range(1, horizon + 1)],
                      columns=[f'quantile_{q}' for q in quantiles])
    for i in range(1, horizon + 1):
        X, Y = generate_current_training_data(DATA, Loading_mapping, horizon=i)
        X = X[X.index == t]
        df = predict_I_value(models[i - 1], X)
        qs.loc[t + pd.Timedelta(hours=i), :] = df[qs.columns].values

    return qs


def predict_oil_future(model_oil, models, DATA, OLMS_top_oil_mapping, Loading_mapping, DGA_mapping, t):
    horizon = len(models)
    Is = predict_Currents(models, DATA, Loading_mapping, horizon, t)
    Xoil, Yoil = generate_training_data_oil(DATA, OLMS_top_oil_mapping, DGA_mapping)
    maxX = maxV[Xoil.columns]
    Xoil = Xoil[(Xoil.index >= t) & (Xoil.index <= t + pd.Timedelta(hours=horizon))]
    OIL_temp = pd.DataFrame(index=[t + pd.Timedelta(hours=h) for h in range(horizon + 1)],
                            columns=['max', 'mean', 'min'])
    OIL_temp.loc[t] = Yoil[t]
    for i in range(1, horizon + 1):
        if i == 1:
            Oil_in = Xoil.loc[t + pd.Timedelta(hours=1)]
            Oil_in['HV Current'] = Is.loc[t + pd.Timedelta(hours=1)].values[0] + 16 * 3
            Oil_in['Top Oil Temperature'] = Yoil[t]
            X = pd.DataFrame(Oil_in.values.reshape(1, Xoil.shape[1]), index=[t + pd.Timedelta(hours=1)],
                             columns=Oil_in.index)
            OIL_temp.loc[t + pd.Timedelta(hours=1), 'max'] = Yoil[t] + 2 * (
                        (model_oil.predict((X / maxX)) * maxX['Top Oil Temperature'])[0][0] - Yoil[t])
            X['HV Current'] = X['HV Current'] - 16 * 3
            OIL_temp.loc[t + pd.Timedelta(hours=1), 'mean'] = Yoil[t] + 2 * (
                        (model_oil.predict((X / maxX)) * maxX['Top Oil Temperature'])[0][0] - Yoil[t])
            X['HV Current'] = X['HV Current'] - 16 * 3
            OIL_temp.loc[t + pd.Timedelta(hours=1), 'min'] = Yoil[t] + 2 * (
                        (model_oil.predict((X / maxX)) * maxX['Top Oil Temperature'])[0][0] - Yoil[t])
        else:
            Oil_in = Xoil.loc[t + pd.Timedelta(hours=i)]
            Oil_in['HV Current'] = Is.loc[t + pd.Timedelta(hours=i)].values[0] + 16 * 3
            Oil_in['Top Oil Temperature'] = OIL_temp.loc[t + pd.Timedelta(hours=i - 1), 'max']
            X = pd.DataFrame(Oil_in.values.reshape(1, Xoil.shape[1]), index=[t + pd.Timedelta(hours=i)],
                             columns=Oil_in.index)
            OIL_temp.loc[t + pd.Timedelta(hours=i), 'max'] = OIL_temp.loc[t + pd.Timedelta(hours=i - 1), 'max'] + 2 * (
                        (model_oil.predict((X / maxX)) * maxX['Top Oil Temperature'])[0][0] - OIL_temp.loc[
                    t + pd.Timedelta(hours=i - 1), 'max'])
            X['HV Current'] = X['HV Current'] - 16 * 3
            X['Top Oil Temperature'] = OIL_temp.loc[t + pd.Timedelta(hours=i - 1), 'mean']
            OIL_temp.loc[t + pd.Timedelta(hours=i), 'mean'] = OIL_temp.loc[
                                                                  t + pd.Timedelta(hours=i - 1), 'mean'] + 2 * ((
                                                                                                                            model_oil.predict(
                                                                                                                                (
                                                                                                                                            X / maxX)) *
                                                                                                                            maxX[
                                                                                                                                'Top Oil Temperature'])[
                                                                                                                    0][
                                                                                                                    0] -
                                                                                                                OIL_temp.loc[
                                                                                                                    t + pd.Timedelta(
                                                                                                                        hours=i - 1), 'mean'])
            X['HV Current'] = X['HV Current'] - 16 * 3
            X['Top Oil Temperature'] = OIL_temp.loc[t + pd.Timedelta(hours=i - 1), 'min']
            OIL_temp.loc[t + pd.Timedelta(hours=i), 'min'] = OIL_temp.loc[t + pd.Timedelta(hours=i - 1), 'min'] + 2 * (
                        (model_oil.predict((X / maxX)) * maxX['Top Oil Temperature'])[0][0] - OIL_temp.loc[
                    t + pd.Timedelta(hours=i - 1), 'min'])

    Probs = probability_to_exceed(60, OIL_temp.loc[OIL_temp.index[1:], 'mean'], (
                OIL_temp.loc[OIL_temp.index[1:], 'max'] - OIL_temp.loc[OIL_temp.index[1:], 'mean']) / 3.4)
    return OIL_temp, Probs * 100


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


# Function to compute the condition state based on H2 value
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


# Function to compute the condition state based on CH4 value
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


# Function to compute the condition state based on C2H4 value
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


# Function to compute the condition state based on C2H6 value
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


# Function to compute the condition state based on C2H2 value
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


def prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping):
    OIL = pd.DataFrame(columns=['Top Oil Temperature', 'Ambient Temperature', 'HV Current'])
    # OLMS_mapping = {'Top Oil Temperature':'Top Oil Temp',
    #                'Ambient Temperature':'Ampient Sun',
    #                 'Ambient Shade Temperature':'Ampient Shade',
    #                 'HV Current':'HV Load Current'}
    for col in OIL.columns:
        print(col)
        print(OLMS_DATA_top_oil_mapping[col])
        print(DATA.loc[DATA.Measurement == OLMS_DATA_top_oil_mapping[col], 'Value'])
        OIL[col] = DATA.loc[DATA.Measurement == OLMS_DATA_top_oil_mapping[col], 'Value'].resample('30min').mean()
    return OIL


def data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping):
    OIL = prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping)
    DGA = prepare_DGA_df(DATA, DGA_mapping)
    ##
    H2 = DGA['H2'].rolling(window=5).mean()
    CH4 = DGA['CH4'].rolling(window=5).mean()
    fig = px.line(H2, x=H2.index, y=H2, title="Sample Scatter Plot")
    fig.add_scatter(x=CH4.index, y=CH4, mode='lines', name='Real', line=dict(color='black'))

    ids = compute_normal_scenarios(DGA)
    OIL = OIL.loc[ids]
    OIL.dropna(inplace=True)
    return OIL


def generate_training_data_oil(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping):
    OIL = data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping)
    train_ids = OIL.index[OIL.rolling('30min').count().sum(axis=1) == OIL.shape[1]]
    train_ids = train_ids[1:]
    Y = OIL.loc[train_ids, 'Top Oil Temperature']
    X = OIL.shift(1).loc[train_ids]
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


def compute_warning_on_DGA(t, DATA, DGA_mapping):
    DGA = prepare_DGA_df(DATA, DGA_mapping)
    DGA_last = DGA[(DGA.index >= (t - pd.Timedelta(days=1))) & (DGA.index <= t)]
    DGA_last['C2H2_State'] = DGA_last['C2H2'].apply(compute_acetylene_condition_state)
    DGA_last['C2H6_State'] = DGA_last['C2H6'].apply(compute_ethane_condition_state)
    DGA_last['C2H4_State'] = DGA_last['C2H4'].apply(compute_ethylene_condition_state)
    DGA_last['CH4_State'] = DGA_last['CH4'].apply(compute_methane_condition_state)
    DGA_last['H2_State'] = DGA_last['H2'].apply(compute_hydrogen_condition_state)
    DGA_last = DGA_last.mean()
    DGA_last['SCORE'] = 50 * DGA_last['H2_State'] + 30 * (
                DGA_last['CH4_State'] + DGA_last['C2H4_State'] + DGA_last['C2H6_State']) + 120 * DGA_last['C2H2_State']
    DGA_last.drop(index=['C2H2_State', 'C2H6_State', 'C2H4_State', 'CH4_State', 'H2_State'], inplace=True)
    DGA = pd.DataFrame(index=['status'], columns=DGA_last.index)
    DGA.loc['status', :] = DGA_last.values.transpose()
    return DGA
