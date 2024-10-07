import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.model_selection import train_test_split
import tf_keras
from tf_keras import layers
import plotly.express as px
import os

maxV = {'Top Oil Temperature':70,'Ambient Temperature':50,'Ambient Shade Temperature':50,'HV Current':300}
maxV =  pd.Series(maxV)


def build_regression_model(input_shape):
    model = tf_keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    return model

def loss_mse(y_true, y_pred):
    e = y_true - y_pred
    return tf.reduce_mean(e*e)

def train_model_top_oil(X_train,y_train):
    maxX = maxV[X_train.columns]
    # Instantiate the model
    input_shape = (X_train.shape[1],)  # Assuming X_train is your feature matrix
    model = build_regression_model(input_shape)
    # Compile the model
    model.compile(optimizer='adam', loss=loss_mse)
    # Train the model
    history = model.fit(X_train / maxX, y_train / maxX['Top Oil Temperature'], epochs=30, verbose=0)
    model.save('Models/Top_Oil')
    return 0

def prepare_model_top_oil(X,Y):
    # Specify the directory path
    directory = "Models/Top_Oil"
    # Check if the directory exists
    if os.path.isdir(directory):
        print("Model exists")
        return 0
    else:
        print("Directory does not exist")
        train_model_top_oil(X, Y)
        return 0



def train_model_bushing_temp_H(X_train,y_train):
    maxX = pd.Series([50,50,50,288,50,50,288],index=X_train.columns)
    # Instantiate the model
    input_shape = (X_train.shape[1],)  # Assuming X_train is your feature matrix
    model = build_regression_model(input_shape)
    # Compile the model
    model.compile(optimizer='adam', loss=loss_mse)
    # Train the model
    history = model.fit(X_train / maxX, y_train / maxX.values[0], epochs=30, verbose=0)
    model.save('Models/Bushing_H')
    return model

def predict_top_oil(X_test,y_test):
    model = tf_keras.models.load_model('Models/Top_Oil',custom_objects={'loss_mse':  loss_mse})
    maxX = maxV[X_test.columns]
    ypred = model.predict(X_test / maxX.values)*maxX['Top Oil Temperature']
    print(loss_mse(y_test,ypred.reshape(ypred.shape[0])))
    print((y_test-ypred.reshape(ypred.shape[0])).max())
    print((y_test - ypred.reshape(ypred.shape[0])).mean())
    print((y_test - ypred.reshape(ypred.shape[0])).std())
    DATA = pd.DataFrame(columns=['Real','Estimate'])
    DATA['Real'] = y_test
    DATA['Estimate'] = ypred
    # # Create a scatter plot
    fig = px.line(DATA, x=DATA.index, y='Estimate', title="Sample Scatter Plot")
    fig.add_scatter(x=DATA.index, y=DATA['Real'], mode='lines', name='Real', line=dict(color='red'))

    # Save the plot as an HTML file
    fig.write_html("scatter_plot.html")

    # Show the plot (optional, will open the plot in a browser)
    fig.show()
    anomaly_detection_in_oil_temp(ypred,y_test,1)
    return 0

def predict_T_bushing(model,X_test,y_test):
    maxX = pd.Series([50,50,50,288,50,50,288],index=X_test.columns)
    ypred = model.predict(X_test / maxX.values)*maxX.values[0]
    print(loss_mse(y_test,ypred.reshape(ypred.shape[0])))
    DATA = pd.DataFrame(columns=['Real','Estimate'])
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
    SCORE = 50*DGA['H2_State']+30*(DGA['CH4_State']+DGA['C2H4_State']+DGA['C2H6_State'])+120*DGA['C2H2_State']
    ids = (SCORE/120)<=3
    return ids

def prepare_DGA_df(DATA):
    DGA = pd.DataFrame(columns=['H2','CH4','C2H2','C2H6','C2H4'])
    OLMS_mapping = {'H2':'TM8 0 H2inOil','CH4':'TM8 0 CH4inOil','C2H2':'TM8 0 C2H2inOil',
                    'C2H6':'TM8 0 C2H6inOil','C2H4':'TM8 0 C2H4inOil'}
    for col in DGA.columns:
        DGA[col] = DATA.loc[DATA.Measurement == OLMS_mapping[col], 'Value'].resample('0.5h').mean()
    #CO2 = ATF3.loc[ATF3.Measurement == 'TM8 0 CO2inOil', 'Value'].resample('0.5h').mean()
    #CO = ATF3.loc[ATF3.Measurement == 'TM8 0 COinOil', 'Value'].resample('0.5h').mean()
    return DGA

def prepare_top_oil_relevant_data(DATA):
    OIL = pd.DataFrame(columns=['Top Oil Temperature','Ambient Temperature','Ambient Shade Temperature','HV Current'])
    OLMS_mapping = {'Top Oil Temperature':'Top Oil Temp',
                   'Ambient Temperature':'Ampient Sun',
                    'Ambient Shade Temperature':'Ampient Shade',
                    'HV Current':'HV Load Current'}
    for col in OIL.columns:
        OIL[col]=DATA.loc[DATA.Measurement == OLMS_mapping[col], 'Value'].resample('0.5h').mean()
    return OIL

def data_cleaning_for_top_oil_train(DATA):
    OIL = prepare_top_oil_relevant_data(DATA)
    DGA = prepare_DGA_df(DATA)
    ##
    H2 = DGA['H2'].rolling(window=5).mean()
    CH4 = DGA['CH4'].rolling(window=5).mean()
    fig = px.line(H2, x=H2.index, y=H2, title="Sample Scatter Plot")
    fig.add_scatter(x=CH4.index, y=CH4, mode='lines', name='Real', line=dict(color='black'))

    # Save the plot as an HTML file
    fig.write_html("DGA_plot.html")
    fig.show()
    ##
    ids = compute_normal_scenarios(DGA)
    OIL = OIL.loc[ids]
    OIL.dropna(inplace=True)
    return OIL

def generate_training_data_oil(DATA):
    OIL = data_cleaning_for_top_oil_train(DATA)
    train_ids = OIL.index[OIL.rolling('0.5h').count().sum(axis=1) == OIL.shape[1]]
    train_ids = train_ids[1:]
    Y = OIL.loc[train_ids,'Top Oil Temperature']
    X = OIL.shift(1).loc[train_ids]
    return X, Y

def anomaly_detection_in_oil_temp(y_pred,y_true,threshold):
    MRi = y_true - pd.Series(y_pred.reshape(y_pred.shape[0]),index=y_true.index)
    MR = MRi.diff().dropna()
    Flags = (MR>=threshold).rolling(window=4).mean() >= 0.75

    # fig = px.line(MR, x=MR.index, y=MR, title="Sample Scatter Plot")
    # #fig.add_scatter(x=MR.index, y=threshold*MR/MR, mode='lines', name='Real', line=dict(color='black'))
    # fig.add_scatter(x=MR.index, y=Flags.astype('float'), mode='lines', name='Real', line=dict(color='black'))
    #
    # # Save the plot as an HTML file
    # fig.write_html("scatter_plot.html")
    # # Show the plot (optional, will open the plot in a browser)
    # fig.show()

    return Flags

def compute_warning_on_bushing(t,DATA):
    Bushings = pd.DataFrame(columns=['Cap H1','Cap H2','Cap H3','Cap Y1','Cap Y2','Cap Y3',
                                'tand H1','tand H2','tand H3','tand Y1','tand Y2','tand Y3'])
    Bushings_mapping = {'Cap H1':'BUSHING H1 Capacitance',
                        'Cap H2':'BUSHING H2 Capacitance',
                        'Cap H3':'BUSHING H3 Capacitance',
                         'Cap Y1':'BUSHING Y1 Capacitance',
                   'Cap Y2':'BUSHING Y2 Capacitance',
                    'Cap Y3':'BUSHING Y3 Capacitance',
                    'tand H1':'BUSHING H1 Tan delta',
                    'tand H2':'BUSHING H2 Tan delta',
                    'tand H3':'BUSHING H3 Tan delta',
                    'tand Y1':'BUSHING Y1 Tan delta',
                    'tand Y2':'BUSHING Y2 Tan delta',
                    'tand Y3':'BUSHING Y3 Tan delta'}
    for col in Bushings.columns:
        Bushings[col] = DATA.loc[DATA.Measurement == Bushings_mapping[col], 'Value'].resample('0.5h').mean()

    Bushings_last = Bushings[(Bushings.index>=(t-pd.Timedelta(days=7)))&(Bushings.index<=t)]
    value = Bushings_last[-6:].mean()
    mean = Bushings_last[:-6].mean()
    abs_change = 100*(mean-value).abs()/mean
    warning = abs_change>=10

    return abs_change[warning]

