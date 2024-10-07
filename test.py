import pandas as pd
import plotly.express as px
from AAM_predict_toolbox import generate_training_data_oil, prepare_model_top_oil, predict_top_oil, compute_warning_on_bushing
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

X, Y = generate_training_data_oil(ATF3)
compute_warning_on_bushing(pd.to_datetime('2024-08-01'),ATF3)

X_train = X[X.index < pd.to_datetime('2024-08-01')]
X_test = X[X.index >= pd.to_datetime('2024-08-01')]
y_train = Y[Y.index < pd.to_datetime('2024-08-01')]
y_test = Y[Y.index >= pd.to_datetime('2024-08-01')]

model = prepare_model_top_oil(X_train,y_train)
predict_top_oil(X_test,y_test)




Cap_H1 = ATF3.loc[ATF3.Measurement=='BUSHING H1 Capacitance','Value'].resample('0.5h').mean()
delta_H1 = ATF3.loc[ATF3.Measurement=='BUSHING H1 Tan delta','Value'].resample('0.5h').mean()
T_H1 = ATF3.loc[ATF3.Measurement=='BUSHING H1 Temperature','Value'].resample('0.5h').mean()
Cap_H2 = ATF3.loc[ATF3.Measurement=='BUSHING H2 Capacitance','Value'].resample('0.5h').mean()
delta_H2 = ATF3.loc[ATF3.Measurement=='BUSHING H2 Tan delta','Value'].resample('0.5h').mean()
T_H2 = ATF3.loc[ATF3.Measurement=='BUSHING H2 Temperature','Value'].resample('0.5h').mean()
Cap_H3 = ATF3.loc[ATF3.Measurement=='BUSHING H3 Capacitance','Value'].resample('0.5h').mean()
delta_H3 = ATF3.loc[ATF3.Measurement=='BUSHING H3 Tan delta','Value'].resample('0.5h').mean()
T_H3 = ATF3.loc[ATF3.Measurement=='BUSHING H3 Temperature','Value'].resample('0.5h').mean()
####
Cap_Y1 = ATF3.loc[ATF3.Measurement=='BUSHING Y1 Capacitance','Value'].resample('0.5h').mean()
delta_Y1 = ATF3.loc[ATF3.Measurement=='BUSHING Y1 Tan delta','Value'].resample('0.5h').mean()
T_Y1 = ATF3.loc[ATF3.Measurement=='BUSHING Y1 Temperature','Value'].resample('0.5h').mean()
Cap_Y2 = ATF3.loc[ATF3.Measurement=='BUSHING Y2 Capacitance','Value'].resample('0.5h').mean()
delta_Y2 = ATF3.loc[ATF3.Measurement=='BUSHING Y2 Tan delta','Value'].resample('0.5h').mean()
T_Y2 = ATF3.loc[ATF3.Measurement=='BUSHING Y2 Temperature','Value'].resample('0.5h').mean()
Cap_Y3 = ATF3.loc[ATF3.Measurement=='BUSHING Y3 Capacitance','Value'].resample('0.5h').mean()
delta_Y3 = ATF3.loc[ATF3.Measurement=='BUSHING Y3 Tan delta','Value'].resample('0.5h').mean()
T_Y3 = ATF3.loc[ATF3.Measurement=='BUSHING Y3 Temperature','Value'].resample('0.5h').mean()






#model = tf_keras.models.load_model('Models/Bushing_H',custom_objects={'loss_mse':  loss_mse})

#predict_T_bushing(model,X_test,y_test)
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
