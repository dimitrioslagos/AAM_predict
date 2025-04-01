import pandas as pd
import plotly.express as px
from AAM_predict_toolbox import train_model_top_oil,anomaly_detection_in_oil_temp, html_future_oil_temp_plot,predict_oil_future,prepare_model_top_oil,generate_current_training_data, train_models_current,predict_Currents,generate_training_data_oil
import tf_keras
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

ATF3 = pd.read_csv('QTMS_Data-2024-09-30_15-55-42_ATF8.csv',delimiter=';')
id1 = ATF3.Timestamp.str.contains('EET')
id2 = ATF3.Timestamp.str.contains('EEST')
T1 = pd.to_datetime(ATF3.loc[id1,'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
T2 = pd.to_datetime(ATF3.loc[id2,'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
T = pd.concat((T1,T2))
ATF3.index = T
ATF3.drop(ATF3.index[ATF3.Logs=='SENSOR ERROR 1'],inplace=True)
ATF3.drop(columns=['Logs'],inplace=True)

# models = train_models_current(ATF3,horizon=6)+
OLMS_DATA_top_oil_mapping = {'Top Oil Temperature': 'Top Oil Temp', 'Ambient Temperature': 'Ampient Sun', 'Ambient Shade Temperature': 'Ampient Shade', 'HV Current': 'HV Load Current'}

DGA_mapping = {'H2': "TM8 0 H2inOil", 'CH4': "TM8 0 CH4inOil", 'C2H2': "TM8 0 C2H2inOil", 'C2H6': "TM8 0 C2H6", 'C2H4': "TM8 0 C2H4"}

Bushings_mapping = {'Cap H1': 'BUSHING H1 Capacitance', 'Cap H2': 'BUSHING H2 Capacitance', 'Cap H3': 'BUSHING H3 Capacitance',
                                         'Cap Y1': 'BUSHING Y1 Capacitance',
                                         'Cap Y2': 'BUSHING Y2 Capacitance',
                                         'Cap Y3': 'BUSHING Y3 Capacitance', 'tand H1': 'BUSHING H1 Tan delta', 'tand H2': 'BUSHING H2 Tan delta', 'tand H3': 'BUSHING H3 Tan delta',
                                         'tand Y1':  'BUSHING Y1 Tan delta', 'tand Y2': 'BUSHING Y2 Tan delta', 'tand Y3': 'BUSHING Y3 Tan delta'}

X, Y = generate_training_data_oil(ATF3, OLMS_DATA_top_oil_mapping, DGA_mapping)
Χ_test=X[X.index.month>=9]
Χ_train=X[X.index.month<8]
Y_test=Y[X.index.month>=9]
Y_train=Y[X.index.month<8]

maxV = {'Top Oil Temperature': 70, 'Ambient Temperature': 50, 'Ambient Shade Temperature': 50, 'HV Current': 300}
maxV = pd.Series(maxV)
model, threshold = prepare_model_top_oil(Χ_train, Y_train)
maxX = maxV[X.columns]
ypred = model.predict(Χ_test / maxX.values) * maxX['Top Oil Temperature']

print(threshold)

print('MSE:',mean_squared_error(Y_test,ypred))
print('MAPE:',mean_absolute_percentage_error(Y_test,ypred))
print('R2:',r2_score(Y_test,ypred))


issues, errors = anomaly_detection_in_oil_temp(ypred, Y_test, threshold)
print(issues.sum())
print(errors)
# t = pd.to_datetime('2024-06-13 11:00:00')
#X =X[X.Measurement == 'HV Load Current']
#X = X['Value'].resample('1h').mean().rename('HV Current')
#Is = predict_Currents(models,ATF3,horizon=6,t)
# OIL_temp = predict_oil_future(model_oil,models,ATF3,t)
# html_future_oil_temp_plot(OIL_temp)
# print('a')
# # # Create a scatter plot
#
# fig = px.scatter(delta_Y1, x=delta_Y1.index, y=delta_Y1, title="Sample Scatter Plot")
#
# # Save the plot as an HTML file
# fig.write_html("scatter_plot3.html")
#
# # Show the plot (optional, will open the plot in a browser)
# fig.show()
#
# fig = px.scatter(Cap_Y1, x=Cap_Y1.index, y=Cap_Y1, title="Sample Scatter Plot")
#
# # Save the plot as an HTML file
# fig.write_html("scatter_plot3.html")
#
# # Show the plot (optional, will open the plot in a browser)
# fig.show()
# #
