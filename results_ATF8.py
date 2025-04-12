import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tf_keras
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from tf_keras import layers
matplotlib.use('TkAgg')

def compute_warning_on_bushing(t, DATA, Bushings_mapping):
    Bushings = pd.DataFrame(columns=['Cap H1', 'Cap H2', 'Cap H3', 'Cap Y1', 'Cap Y2', 'Cap Y3',
                                     'tand H1', 'tand H2', 'tand H3', 'tand Y1', 'tand Y2', 'tand Y3'])
    for col in Bushings.columns:
        Bushings[col] = DATA.loc[DATA.Measurement == Bushings_mapping[col], 'Value'].resample('30min').mean()

    Bushings_last = Bushings[(Bushings.index >= (t - pd.Timedelta(days=7))) & (Bushings.index <= t)]
    value = Bushings_last[-6:].mean()
    mean = Bushings_last[:-6].mean()
    abs_change = 100 * (mean - value).abs() / mean
    warning = abs_change >= 3
    Message = pd.DataFrame(columns=warning.index)
    Message.loc[0, Message.columns[warning]] = 'Warning'
    Message.loc[0, Message.columns[warning == False]] = 'OK'
    Message.index = ['Status']
    return Message

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

def prepare_DGA_df(DATA, OLMS_mapping):
    DGA = pd.DataFrame(columns=['H2', 'CH4', 'C2H2', 'C2H6', 'C2H4'])
    for col in DGA.columns:
        DGA[col] = DATA.loc[DATA.Measurement == OLMS_mapping[col], 'Value'].resample('30min').mean()

    return DGA

def prepare_Bushing_df(DATA, Bushings_mapping):
    Bushings = pd.DataFrame(columns=['Cap H1', 'Cap H2', 'Cap H3', 'Cap Y1', 'Cap Y2', 'Cap Y3',
                                     'tand H1', 'tand H2', 'tand H3', 'tand Y1', 'tand Y2', 'tand Y3'])
    for col in Bushings.columns:
        Bushings[col] = DATA.loc[DATA.Measurement == Bushings_mapping[col], 'Value'].resample('30min').mean()
    return Bushings

def compute_normal_scenarios(DGA):
    DGA['C2H2_State'] = DGA['C2H2'].apply(compute_acetylene_condition_state)
    DGA['C2H6_State'] = DGA['C2H6'].apply(compute_ethane_condition_state)
    DGA['C2H4_State'] = DGA['C2H4'].apply(compute_ethylene_condition_state)
    DGA['CH4_State'] = DGA['CH4'].apply(compute_methane_condition_state)
    DGA['H2_State'] = DGA['H2'].apply(compute_hydrogen_condition_state)
    SCORE = 50 * DGA['H2_State'] + 30 * (DGA['CH4_State'] + DGA['C2H4_State'] + DGA['C2H6_State']) + 120 * DGA[
        'C2H2_State']
    condition = (SCORE / 120) <= 3

    # Return the Series with the index as DatetimeIndex and the boolean values
    ids = pd.Series(condition, index=DGA.index)

    return ids

def data_find_abnormal_DGA(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping):
    OIL = prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping).dropna()
    DGA = prepare_DGA_df(DATA, DGA_mapping).dropna()
    ids = compute_normal_scenarios(DGA)
    ids_not_normal = ids[ids == False].index

    return ids_not_normal

def plot_anomalies(anomalies, MR, MRmean, UCL, LCL, output_dir="bushing_plots_ATF8"):
    os.makedirs(output_dir, exist_ok=True)
    for idx, anomaly in anomalies.iterrows():
        anomaly_start_timestamp = idx
        anomaly_date = anomaly_start_timestamp
        single_day_data = MR.loc[str(anomaly_date)]
        plt.figure(figsize=(10, 6))
        plt.plot(single_day_data.index, single_day_data, label="MR Values", color='blue', alpha=0.7, marker='o')
        plt.axhline(MRmean, color='green', linestyle='--', label="Mean of MR (Mean)")
        plt.axhline(UCL, color='red', linestyle='--', label="UCL (Upper Control Limit)")
        plt.axhline(LCL, color='orange', linestyle='--', label="LCL (Lower Control Limit)")
        plt.fill_between(single_day_data.index, LCL, UCL, color='gray', alpha=0.2, label="Control Band")
        plt.title(f'Mean, UCL, LCL, and MR Values for {anomaly_date}', fontsize=16)
        plt.xlabel('Timestamp', fontsize=14)
        plt.ylabel('MR Value', fontsize=14)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plot_filename = f"{output_dir}/anomaly_{anomaly_date}.png"
        plt.savefig(plot_filename)
        plt.close()

def prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping):
    OIL = pd.DataFrame(columns=['Top Oil Temperature', 'Ambient Temperature', 'HV Current'])
    for col in OIL.columns:
        OIL[col] = DATA.loc[DATA.Measurement == OLMS_DATA_top_oil_mapping[col], 'Value'].resample('30min').mean()
    return OIL

def quantile_90_bins(OIL, current_bins, temperature_bins):
    mask = pd.Series(False, index=OIL.index)

    for i in range(len(current_bins) - 1):
        for j in range(len(temperature_bins) - 1):
            bin_data = OIL[
                (OIL['HV Current'] >= current_bins[i]) &
                (OIL['HV Current'] < current_bins[i + 1]) &
                (OIL['Ambient Temperature'] >= temperature_bins[j]) &
                (OIL['Ambient Temperature'] < temperature_bins[j + 1])
                ]

            if not bin_data.empty:  # If bin_data has any rows
                Temp_quantile_005 = bin_data['Top Oil Temperature'].quantile(0.05)
                Temp_quantile_095 = bin_data['Top Oil Temperature'].quantile(0.95)

                # Filter the bin_data to keep values within quantiles
                bin_data_filtered = bin_data[
                    (bin_data['Top Oil Temperature'] >= Temp_quantile_005) &
                    (bin_data['Top Oil Temperature'] <= Temp_quantile_095)
                    ]

                # Mark the rows in mask as True if they are part of the valid filtered data
                mask.loc[bin_data_filtered.index] = True

    return OIL, mask

def data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping):
    OIL = prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping)
    maxT=10 * np.ceil(OIL['Ambient Temperature'].max()/ 10)
    minT=10 * np.floor(OIL['Ambient Temperature'].min()/ 10)
    maxI=5 * np.ceil(OIL['HV Current'].max()/ 5)
    minI=5 * np.floor(OIL['HV Current'].min()/ 5)
    current_bins = np.arange(minI, maxI + 5, 5)
    temperature_bins = np.arange(minT, maxT + 10, 10)
    OIL, mask = quantile_90_bins(OIL, current_bins, temperature_bins)
    OIL.dropna(inplace=True)
    mask = mask.loc[OIL.index]
    return OIL, mask

def generate_training_data_oil(DATA, OLMS_DATA_top_oil_mapping):
    OIL, mask = data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping)

    # Raw X and Y (without using mask)
    train_ids = OIL.index[OIL.rolling('30min').count().sum(axis=1) == OIL.shape[1]]
    train_ids = train_ids[1:]
    Y_raw = OIL.loc[train_ids, 'Top Oil Temperature']
    X_raw = OIL.shift(1).loc[train_ids]

    # Filtered X and Y using the mask
    Y_filtered = Y_raw[mask.loc[Y_raw.index]]
    X_filtered = X_raw[mask.loc[X_raw.index]]

    return X_raw, Y_raw, X_filtered, Y_filtered

def loss_mse(y_true, y_pred):
    e = y_true - y_pred
    return tf.reduce_mean(e * e)

def build_regression_model(input_shape):
    model = tf_keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    return model

def train_model_top_oil(X_train, y_train):
    maxX = maxV[X_train.columns]
    input_shape = (X_train.shape[1],)  # Assuming X_train is your feature matrix
    model = build_regression_model(input_shape)
    model.compile(optimizer='adam', loss=loss_mse)
    model.fit(X_train / maxX, y_train / maxX['Top Oil Temperature'], epochs=50, verbose=0)
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

def anomaly_detection_in_oil_temp(y_pred, y_true, window):
    MRi = y_true - y_pred
    MR = MRi.diff().abs().dropna()
    MRmean = MR.mean()
    MRstd = MR.std()
    Flags = ((MR-MRmean) >= MRstd).rolling(window=4).mean() >= 0.75
    UCL = MRmean + 3*MRstd
    LCL = MRmean - 3*MRstd
    consecutive_above_ucl = 0
    consecutive_below_lcl = 0
    anomalies = []

    for i in range(window, len(MR)):
        # Check if the deviation is above UCL or below LCL
        if MR.iloc[i] > UCL:
            consecutive_above_ucl += 1
            consecutive_below_lcl = 0  # Reset below LCL counter
        elif MR.iloc[i] < LCL:
            consecutive_below_lcl += 1
            consecutive_above_ucl = 0  # Reset above UCL counter
        else:
            consecutive_above_ucl = 0
            consecutive_below_lcl = 0

        if consecutive_above_ucl >= 2 or consecutive_below_lcl >= 2:
            if i > 0 and (MR.index[i] - MR.index[i - 1]).total_seconds() > 1800:
                # If the gap is more than 30 minutes, reset the counters and skip
                consecutive_above_ucl = 0
                consecutive_below_lcl = 0
                continue

        # If there are 4 consecutive intervals above UCL or below LCL, flag it as an anomaly
        if consecutive_above_ucl >= window or consecutive_below_lcl >= window:
            anomaly_start_timestamp = MR.index[i - window +1]  # The timestamp of the first anomaly in the range
            anomaly_end_timestamp = MR.index[i]  # The timestamp of the last anomaly in the range
            message = f"Anomaly detected from {anomaly_start_timestamp} to {anomaly_end_timestamp}"
            anomalies.append((anomaly_start_timestamp, message))
    anomalies = pd.DataFrame(anomalies, columns=["Start Timestamp", "Message"])
    anomalies["Start Timestamp"] = anomalies["Start Timestamp"].dt.date
    anomalies.set_index("Start Timestamp", inplace=True)

    return anomalies, MR, MRmean, UCL, LCL, Flags

ATF8 = pd.read_csv('QTMS_Data-2024-09-30_15-55-42_ATF8.csv', delimiter=';', low_memory=False)
id1 = ATF8.Timestamp.str.contains('EET')
id2 = ATF8.Timestamp.str.contains('EEST')
T1 = pd.to_datetime(ATF8.loc[id1, 'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
T2 = pd.to_datetime(ATF8.loc[id2, 'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
T = pd.concat((T1, T2))
ATF8.index = T
ATF8.drop(ATF8.index[ATF8.Logs=='SENSOR ERROR 1'], inplace=True)
ATF8.drop(columns=['Logs'], inplace=True)

OLMS_DATA_top_oil_mapping = {'Top Oil Temperature': 'Top Oil Temp', 'Ambient Temperature': 'Ampient Sun', 'Ambient Shade Temperature': 'Ampient Shade', 'HV Current': 'HV Load Current'}

DGA_mapping = {'H2': "TM8 0 H2inOil", 'CH4': "TM8 0 CH4inOil", 'C2H2': "TM8 0 C2H2inOil", 'C2H6': "TM8 0 C2H6inOil", 'C2H4': "TM8 0 C2H4inOil"}

Bushings_mapping = {'Cap H1': 'BUSHING H1 Capacitance', 'Cap H2': 'BUSHING H2 Capacitance', 'Cap H3': 'BUSHING H3 Capacitance',
                    'Cap Y1': 'BUSHING Y1 Capacitance',
                    'Cap Y2': 'BUSHING Y2 Capacitance',
                    'Cap Y3': 'BUSHING Y3 Capacitance', 'tand H1': 'BUSHING H1 Tan delta', 'tand H2': 'BUSHING H2 Tan delta', 'tand H3': 'BUSHING H3 Tan delta',
                    'tand Y1':  'BUSHING Y1 Tan delta', 'tand Y2': 'BUSHING Y2 Tan delta', 'tand Y3': 'BUSHING Y3 Tan delta'}

X, Y, X_filtered, Y_filtered = generate_training_data_oil(ATF8, OLMS_DATA_top_oil_mapping)
X_test=X[X.index.month>=8]
Y_test=Y[X.index.month>=8]
X_train=X_filtered[X_filtered.index.month<8]
Y_train=Y_filtered[X_filtered.index.month<8]

maxV = {'Top Oil Temperature': 70, 'Ambient Temperature': 50, 'Ambient Shade Temperature': 50, 'HV Current': 300}
maxV = pd.Series(maxV)
model, threshold = prepare_model_top_oil(X_train, Y_train)
maxX = maxV[X_test.columns]
ypred = model.predict(X_test / maxX.values) * maxX['Top Oil Temperature']
ypred = ypred.flatten()
ypred = pd.Series(ypred, index=Y_test.index)

print('threshold:', threshold)
print('MSE:', mean_squared_error(Y_test, ypred))
print('MAPE:', mean_absolute_percentage_error(Y_test, ypred))
print('R2:', r2_score(Y_test, ypred))

#find anomalies with the model
anomalies, MR, MRmean, UCL, LCL, Flags = anomaly_detection_in_oil_temp(ypred, Y_test, 3)
plot_anomalies(anomalies, MR, MRmean, UCL, LCL)

#find ids marked abnormal by DGA
ids = data_find_abnormal_DGA(ATF8, OLMS_DATA_top_oil_mapping, DGA_mapping)
anomalies_from_DGA=pd.DataFrame(ids)
anomalies_from_DGA.set_index("Timestamp", inplace=True)

#Run warning bushing for all test days
start_date = pd.to_datetime('2024-08-01')
unique_dates = pd.to_datetime(ATF8.index.date).unique()
unique_dates = unique_dates[unique_dates >= start_date]
warnings_over_time = []
for current_day in unique_dates:
    current_day_ts = pd.to_datetime(current_day)
    alarm = compute_warning_on_bushing(current_day_ts, ATF8, Bushings_mapping)
    alarm["Date"] = current_day_ts.date()  # Add date as column
    warnings_over_time.append(alarm)

# Combine all warnings into a single DataFrame
Alarms_all = pd.concat(warnings_over_time).set_index("Date")


Bushing = prepare_Bushing_df(ATF8, Bushings_mapping)
output_dir = "bushing_plots_ATF8"
os.makedirs(output_dir, exist_ok=True)
for date in anomalies.index:
    bushing_day_measurements = Bushing.loc[Bushing.index.date == date]

    if not bushing_day_measurements.empty:
        # Define plot groups
        cap_y_cols = [col for col in bushing_day_measurements.columns if "Cap" in col and "Y" in col]
        cap_h_cols = [col for col in bushing_day_measurements.columns if "Cap" in col and "H" in col]
        tand_cols  = [col for col in bushing_day_measurements.columns if "tand" in col]

        # Plot Cap Y
        plt.figure(figsize=(10, 4))
        for col in cap_y_cols:
            plt.plot(bushing_day_measurements.index, bushing_day_measurements[col], label=col)
        plt.title(f'Capacitance Y Bushings for {date}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'Cap_Y_{date}.png')
        plt.savefig(filename)
        plt.close()

        # Plot Cap H
        plt.figure(figsize=(10, 4))
        for col in cap_h_cols:
            plt.plot(bushing_day_measurements.index, bushing_day_measurements[col], label=col)
        plt.title(f'Capacitance H Bushings for {date}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'Cap_H_{date}.png')
        plt.savefig(filename)
        plt.close()

        # Plot Tan Delta
        plt.figure(figsize=(10, 4))
        for col in tand_cols:
            plt.plot(bushing_day_measurements.index, bushing_day_measurements[col], label=col)
        plt.title(f'Tan Delta Bushings for {date}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'TanD_{date}.png')
        plt.savefig(filename)
        plt.close()


print('a')
