import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tf_keras
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tf_keras import layers

def prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping):
    OIL = pd.DataFrame(columns=['Top Oil Temperature', 'Ambient Temperature', 'HV Current'])
    for col in OIL.columns:
        OIL[col] = DATA.loc[DATA.Measurement == OLMS_DATA_top_oil_mapping[col], 'Value'].resample('30min').mean()
    return OIL


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

def data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping):
    OIL = prepare_top_oil_relevant_data(DATA, OLMS_DATA_top_oil_mapping)
    maxT=10 * np.ceil(OIL['Ambient Temperature'].max()/ 10)
    minT=10 * np.floor(OIL['Ambient Temperature'].min()/ 10)
    maxI=5 * np.ceil(OIL['HV Current'].max()/ 5)
    minI=5 * np.floor(OIL['HV Current'].min()/ 5)
    current_bins = np.arange(minI, maxI + 5, 5)
    temperature_bins = np.arange(minT, maxT + 10, 10)
    OIL_filtered = quantile_90_bins(OIL, current_bins, temperature_bins)
    OIL_filtered.dropna(inplace=True)
    return OIL_filtered


def generate_training_data_oil(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping):
    OIL = data_cleaning_for_top_oil_train(DATA, OLMS_DATA_top_oil_mapping, DGA_mapping)
    train_ids = OIL.index[OIL.rolling('30min').count().sum(axis=1) == OIL.shape[1]]
    train_ids = train_ids[1:]
    Y = OIL.loc[train_ids, 'Top Oil Temperature']
    X = OIL.shift(1).loc[train_ids]
    return X, Y

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
    # Instantiate the model
    input_shape = (X_train.shape[1],)  # Assuming X_train is your feature matrix
    model = build_regression_model(input_shape)
    # Compile the model
    model.compile(optimizer='adam', loss=loss_mse)
    # Train the model
    history = model.fit(X_train / maxX, y_train / maxX['Top Oil Temperature'], epochs=50, verbose=0)
    # model.save('Models/Top_Oil')
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

def anomaly_detection_in_oil_temp(y_pred, y_true):
    MRi = y_true - y_pred
    MR = MRi.diff().abs().dropna()
    #Flags = (MR >= threshold).rolling(window=4).mean() >= 0.75
    MRmean = MR.mean()
    MRstd = MR.std()
    UCL = MRmean + 3*MRstd
    LCL = MRmean - 3*MRstd
    consecutive_above_ucl = 0
    consecutive_below_lcl = 0
    anomalies = []

    for i in range(len(MR)):
        # Check if the deviation is above UCL or below LCL
        if MR[i] > UCL:
            consecutive_above_ucl += 1
            consecutive_below_lcl = 0  # Reset below LCL counter
        elif MR[i] < LCL:
            consecutive_below_lcl += 1
            consecutive_above_ucl = 0  # Reset above UCL counter
        else:
            consecutive_above_ucl = 0
            consecutive_below_lcl = 0

        # If there are 4 consecutive intervals above UCL or below LCL, flag it as an anomaly
        if consecutive_above_ucl >= 3 or consecutive_below_lcl >= 3:
            anomaly_start_timestamp = MR.index[i - 3]  # The timestamp of the first anomaly in the range
            anomaly_end_timestamp = MR.index[i]  # The timestamp of the last anomaly in the range
            anomalies.append(f"Anomaly detected from {anomaly_start_timestamp} to {anomaly_end_timestamp}")

    # Filter data for 30th August
    single_day_data = MR.loc['2024-09-10']

    # Plotting the MR, UCL, LCL, and MR mean for the 30th August
    plt.figure(figsize=(10, 6))

    # Plot MR values for the day (just one value in this case)
    plt.plot(single_day_data.index, single_day_data, label="MR Values", color='blue', alpha=0.7, marker='o')

    # Plot Mean of MR
    plt.axhline(MRmean, color='green', linestyle='--', label="Mean of MR (Mean)")

    # Plot Upper Control Limit (UCL)
    plt.axhline(UCL, color='red', linestyle='--', label="UCL (Upper Control Limit)")

    # Plot Lower Control Limit (LCL)
    plt.axhline(LCL, color='orange', linestyle='--', label="LCL (Lower Control Limit)")

    # Fill between LCL and UCL to indicate the control band area
    plt.fill_between([single_day_data.index[0], single_day_data.index[0]], LCL, UCL, color='gray', alpha=0.2,
                     label="Control Band")

    # Adding labels, legend, and title
    plt.title('Mean, UCL, LCL, and MR Values for 2024-09-10', fontsize=16)
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('MR Value', fontsize=14)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True)

    # Increase the font size of tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    # Display the plot
    plt.show()

    return anomalies, MR

ATF8 = pd.read_csv('QTMS_Data-2024-09-30_15-55-42_ATF8.csv',delimiter=';')
id1 = ATF8.Timestamp.str.contains('EET')
id2 = ATF8.Timestamp.str.contains('EEST')
T1 = pd.to_datetime(ATF8.loc[id1,'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
T2 = pd.to_datetime(ATF8.loc[id2,'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
T = pd.concat((T1,T2))
ATF8.index = T
ATF8.drop(ATF8.index[ATF8.Logs=='SENSOR ERROR 1'],inplace=True)
ATF8.drop(columns=['Logs'],inplace=True)

# models = train_models_current(ATF3,horizon=6)+
OLMS_DATA_top_oil_mapping = {'Top Oil Temperature': 'Top Oil Temp', 'Ambient Temperature': 'Ampient Sun', 'Ambient Shade Temperature': 'Ampient Shade', 'HV Current': 'HV Load Current'}

DGA_mapping = {'H2': "TM8 0 H2inOil", 'CH4': "TM8 0 CH4inOil", 'C2H2': "TM8 0 C2H2inOil", 'C2H6': "TM8 0 C2H6", 'C2H4': "TM8 0 C2H4"}

Bushings_mapping = {'Cap H1': 'BUSHING H1 Capacitance', 'Cap H2': 'BUSHING H2 Capacitance', 'Cap H3': 'BUSHING H3 Capacitance',
                                         'Cap Y1': 'BUSHING Y1 Capacitance',
                                         'Cap Y2': 'BUSHING Y2 Capacitance',
                                         'Cap Y3': 'BUSHING Y3 Capacitance', 'tand H1': 'BUSHING H1 Tan delta', 'tand H2': 'BUSHING H2 Tan delta', 'tand H3': 'BUSHING H3 Tan delta',
                                         'tand Y1':  'BUSHING Y1 Tan delta', 'tand Y2': 'BUSHING Y2 Tan delta', 'tand Y3': 'BUSHING Y3 Tan delta'}

ATF3 = pd.read_csv('atf_3_from_2024-09-30_data.csv')
ATF3['Date'] = pd.to_datetime(ATF3['Date'])
ATF3=ATF3.sort_values(by=['Date', 'ID'])
ATF3.index = ATF3['Date']
ATF3.drop(ATF3.index[ATF3.Logs=='SENSOR ERROR 1'], inplace=True)
ATF3.drop(columns=['Logs'], inplace=True)

X, Y = generate_training_data_oil(ATF8, OLMS_DATA_top_oil_mapping, DGA_mapping)
X_test=X[X.index.month>=8]
Y_test=Y[X.index.month>=8]
X_train=X[X.index.month<8]
Y_train=Y[X.index.month<8]

maxV = {'Top Oil Temperature': 70, 'Ambient Temperature': 50, 'Ambient Shade Temperature': 50, 'HV Current': 300}
maxV = pd.Series(maxV)
model, threshold = prepare_model_top_oil(X_train, Y_train)
maxX = maxV[X.columns]
ypred = model.predict(X_test / maxX.values) * maxX['Top Oil Temperature']
ypred = ypred.flatten()
ypred = pd.Series(ypred, index=Y_test.index)

print(threshold)
print('MSE:',mean_squared_error(Y_test,ypred))
print('MAPE:',mean_absolute_percentage_error(Y_test,ypred))
print('R2:',r2_score(Y_test,ypred))

issues, errors = anomaly_detection_in_oil_temp(ypred, Y_test)
print(issues)
print(errors)

print('a')
