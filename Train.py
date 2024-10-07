import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tf_keras
from quantile_forest import RandomForestQuantileRegressor
from tf_keras import layers


import plotly.graph_objs as go
import plotly.io as pio


tfd = tfp.distributions


def plot(y_test, y_pred):
    layout = go.Layout(
        legend=dict(
            orientation='h',  # Horizontal orientation
            yanchor='bottom',  # Anchoring to the bottom of the legend
            y=1.0,  # Position the legend above the plot area
            xanchor='center',  # Center the legend horizontally
            x=1.0  # Center the legend horizontally
        ),
        title=dict(
            text='Line Loading Forecast',  # Title text
            x=0.5,  # X-position: 0.5 means centered
            y=1.0,  # Y-position: adjust as needed
            xanchor='center',  # Anchor the title at the center
            yanchor='top'  # Anchor the title at the top
        ),
        xaxis=dict(title='Time'),
        yaxis=dict(title='Vals (%)',
                   tickmode='array', ))
    # Define which y-axis ticks to show

    fig = go.Figure(layout=layout)

    fig.add_traces(go.Scatter(
        x=np.array(range(y_test.shape[0])),
        y=y_test,
        mode='markers',
        marker=dict(color='red'),
        name='actual',
        showlegend=False
    ))
    fig.add_traces(go.Scatter(
        x=np.array(range(y_test.shape[0])),
        y=y_pred[:, 0],
        mode='markers',
        marker=dict(color='blue'),
        name='50',
        showlegend=False
    ))

    fig.add_traces(go.Scatter(
        x=np.array(range(y_test.shape[0])),
        y=y_pred[:, 1],
        mode='markers',
        marker=dict(color='blue'),
        name='70',
        showlegend=False
    ))

    fig.add_traces(go.Scatter(
        x=np.array(range(y_test.shape[0])),
        y=y_pred[:, 2],
        mode='markers',
        marker=dict(color='blue'),
        name='90',
        showlegend=False
    ))

    fig.add_traces(go.Scatter(
        x=np.array(range(y_test.shape[0])),
        y=y_pred[:, 3],
        mode='markers',
        marker=dict(color='blue'),
        name='99',
        showlegend=False
    ))


    # Save the figure as an HTML file
    pio.write_html(fig, file='loading_plot.html', auto_open=True)
    return 0

def pinball_loss(q, y_true, y_pred):
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

def pinball_loss_up(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), axis=-1)
    return loss


def build_quantile_regression_model(input_shape):
    model = tf_keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    return model


FILES = ['DGA_AG_ST.csv','Pallini_ATF_6_DGA.csv','Pallini_ATF_8_DGA.csv','Pallini_TF3.csv']

GASES = ['Hydrogen (H2)', 'Oxygen (O2)', 'Methane (CH4)',
         'Carbon Monoxide (CO)', 'Carbon Dioxide(CO2)', 'Ethane (C2H6)',
         'Ethylene (C2H4)', 'Acetylene(C2H2)', 'Nitrogen(N2)']
HORIZON_BACK = 7
GASES_PREDICT = ['Hydrogen (H2)', 'Methane (CH4)',
         'Carbon Monoxide (CO)', 'Carbon Dioxide(CO2)', 'Ethane (C2H6)',
         'Ethylene (C2H4)', 'Acetylene(C2H2)']
GASES_min = pd.DataFrame(index=GASES,columns=FILES)
##GASmin = {'Hydrogen (H2)':3,}
for gas in GASES:
       GASES = ['Hydrogen (H2)', 'Oxygen (O2)', 'Methane (CH4)',
              'Carbon Monoxide (CO)', 'Carbon Dioxide(CO2)', 'Ethane (C2H6)',
              'Ethylene (C2H4)', 'Acetylene(C2H2)', 'Nitrogen(N2)']
       for file in FILES:
              DATA_AS = pd.read_csv(file)
              GASES_min.loc[gas,file] = DATA_AS[gas].min()
GAS_MIN = GASES_min.min(axis=1)
###Clean Data
DATA_AS = pd.read_csv('DGA_AG_ST.csv',index_col=0)
DATA_AS.index = pd.DatetimeIndex(DATA_AS.index )
ids = DATA_AS[GASES].isna().sum(axis=1)<9
DATA = DATA_AS.loc[ids]
DATA = DATA.resample('1D').mean()
train_ids = DATA.index[DATA['Oxygen (O2)'].rolling(str(HORIZON_BACK+1)+'D').count()==HORIZON_BACK+1]
##Scaling
DATA = DATA[GASES]
for gas in GASES:
    DATA.loc[DATA[gas].isna(), gas] = GAS_MIN[gas]
#scaler = StandardScaler()
#DATA2 = scaler.fit_transform(DATA)
#DATA2 = pd.DataFrame(DATA2,index=DATA.index,columns=DATA.columns)
#
DATA2 = DATA
X=DATA2.loc[train_ids]


TARGETS = pd.DataFrame(index=X.index)
TARGETS = X[GASES_PREDICT]
TRAIN = pd.DataFrame(index=X.index)
for i in TRAIN.index:
    for g in GASES:
        for h in range(1,HORIZON_BACK+1):
            TRAIN.loc[i,g+'_'+str(h)] = DATA2.loc[i-pd.Timedelta(days=h),g]

#Scaling
days = [3,6,9,12,15]
for gas in GASES_PREDICT:
    for day in days:
        #Splitting
        # Assuming you have input features X and target labels y
        print(TARGETS[gas].std())
        if TARGETS[gas].std()<=5:
            continue
        ids = []
        for i in TRAIN.index:
            if ((TRAIN.index<=(i+pd.Timedelta(days=day)))&(TRAIN.index>=i)).sum()==day+1:
                ids.append(i)

        X_train, X_test, y_train, y_test = train_test_split(TRAIN.loc[ids], TARGETS[gas].shift(periods=-day).loc[ids], test_size=0.7, random_state=42)
        X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, test_size=0.15, random_state=42)



        qrf = RandomForestQuantileRegressor(n_estimators=100,criterion='squared_error',
                                                n_jobs=2,min_samples_leaf=10, default_quantiles=[0.5, 0.7, 0.9, 0.99])
        qrf.fit(X_train, y_train)



        y_pred = qrf.predict(X_test, quantiles=[0.5, 0.7, 0.9, 0.99])
        #plot(y_test, y_pred)

        print(pinball_loss(0.5,y_test, y_pred[:,0])+pinball_loss(0.7,y_test, y_pred[:,1])
              +pinball_loss(0.9,y_test, y_pred[:,2])+pinball_loss(0.99,y_test, y_pred[:,3]))# maxX = X_train.max(axis=0)
        #
        X_train2 = X_train.loc[:,X_train.columns[qrf.feature_importances_ >= 1e-2]]
        X_test2 = X_test.loc[:,X_test.columns[qrf.feature_importances_ >= 1e-2]]
        X_eval2 = X_eval.loc[:,X_eval.columns[qrf.feature_importances_ >= 1e-2]]
        #print(X_train2.shape[1])
        Qs = [0.5,0.7,0.9,0.99]
        res = 0
        #columns = pd.read_csv('Models/AG_ST/'+'column_names.csv')
        #X_test2 = X_test[columns.columns]
        #X_train2 = X_train[columns.columns]
        pd.DataFrame([X_train2.columns.tolist()]).to_csv('Models/AG_ST/'+str(gas)+'_'+str(day)+'_column_names.csv',
                                                         header=False, index=False)
        for q in Qs:
            maxX = X_train2.max(axis=0)
            # Instantiate the model
            input_shape = (X_train2.shape[1],)  # Assuming X_train is your feature matrix
            model = build_quantile_regression_model(input_shape)
            # Compile the model
            model.compile(optimizer='adam', loss=pinball_loss_up(q))
            # Train the model
            history = model.fit(X_train2/maxX, y_train/maxX.values[0], epochs=200, verbose=0,
                                validation_data=(X_eval2/maxX, y_eval/maxX.values[0]),)
            # Predict using the trained model
            model.save('Models/AG_ST/'+gas+'_'+str(day)+'_Q'+str(q))
            y_preda = model.predict(X_test2/maxX.values)
            y_mean1 = y_preda*(maxX.values[0])
            y_pred[:,Qs.index(q)] = y_mean1.reshape(y_mean1.shape[0])
            res = res + pinball_loss(q, y_test.values.reshape(y_test.shape[0], 1), y_mean1)
        print(res)
        plot(y_test, y_pred)
# for q in Qs:
#     maxX = X_train2.max(axis=0)
#     loaded_model = tf_keras.models.load_model(
#         'models/AG_ST/'+GASES[0]+'_'+str(1)+'_Q'+str(q),
#         custom_objects={'loss':  pinball_loss_up(q)}
#     )
#     y_preda = loaded_model.predict(X_test2/maxX.values)
#     y_mean1 = y_preda*(maxX.values[0])
#     y_pred[:,Qs.index(q)] = y_mean1.reshape(y_mean1.shape[0])
#
#
#
# plot(y_test, y_pred)
