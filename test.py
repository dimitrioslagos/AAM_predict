import pandas as pd
import plotly.express as px
from AAM_predict_toolbox import html_future_oil_temp_plot,predict_oil_future,prepare_model_top_oil,generate_current_training_data, train_models_current,predict_Currents,generate_training_data_oil
import tf_keras

ATF3 = pd.read_csv('QTMS_Data-2024-09-30_15-55-42_ATF8.csv',delimiter=';')
id1 = ATF3.Timestamp.str.contains('EET')
id2 = ATF3.Timestamp.str.contains('EEST')
T1 = pd.to_datetime(ATF3.loc[id1,'Timestamp'], format='%m/%d/%y, %H:%M:%S EET')
T2 = pd.to_datetime(ATF3.loc[id2,'Timestamp'], format='%m/%d/%y, %H:%M:%S EEST')
T = pd.concat((T1,T2))
ATF3.index = T
ATF3.drop(ATF3.index[ATF3.Logs=='SENSOR ERROR 1'],inplace=True)
ATF3.drop(columns=['Logs'],inplace=True)

models = train_models_current(ATF3,horizon=6)

X, Y = generate_training_data_oil(ATF3)
model_oil, oil_threshold = prepare_model_top_oil(X, Y)
t = pd.to_datetime('2024-06-13 11:00:00')
#X =X[X.Measurement == 'HV Load Current']
#X = X['Value'].resample('1h').mean().rename('HV Current')
#Is = predict_Currents(models,ATF3,horizon=6,t)
OIL_temp = predict_oil_future(model_oil,models,ATF3,t)
html_future_oil_temp_plot(OIL_temp)
print('a')
# # Create a scatter plot

fig = px.scatter(delta_Y1, x=delta_Y1.index, y=delta_Y1, title="Sample Scatter Plot")

# Save the plot as an HTML file
fig.write_html("scatter_plot3.html")

# Show the plot (optional, will open the plot in a browser)
fig.show()

fig = px.scatter(Cap_Y1, x=Cap_Y1.index, y=Cap_Y1, title="Sample Scatter Plot")

# Save the plot as an HTML file
fig.write_html("scatter_plot3.html")

# Show the plot (optional, will open the plot in a browser)
fig.show()
#
