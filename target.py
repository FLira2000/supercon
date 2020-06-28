print("Note: This computation may take a while to be performed. Please wait. ")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from thermo import simple_formula_parser
import datetime
from mail import mailSender

agora = datetime.datetime.now()
mailMessage = str(agora.strftime("%Y-%m-%d %H.%M")) + "\n \n"
mailConfig = { "email": "data.supercon@gmail.com", "password": "rnzizqregtssljji" }
PLOT_IMAGE_NAME = "plot_" + agora.strftime("%Y-%m-%d_%H:%M") + ".png"

#importing the dataframe
df = pd.read_csv('ybaco_materials_all.csv')
df.head()

#select a superconductor to further testing

ybaco7 = df.loc[df['Unnamed: 0'] == 29]
ybaco7 = ybaco7.drop(['material'], axis=1)
observedValue = np.array(ybaco7['critical_temp'])

print('\nUsing Y1Ba2Cu3O7 for testing predictions, observed value: ' + str(observedValue))
mailMessage += '\nUsing Y1Ba2Cu3O7 for testing predictions, observed value: ' + str(observedValue)

ybaco7 = ybaco7.drop(['critical_temp'], axis=1)
ybaco7 = ybaco7.drop(['Unnamed: 0'], axis = 1).drop(['Unnamed: 0.1'], axis = 1).select_dtypes(exclude=['object'])
ybaco7 = np.array(ybaco7)

#linear definition
y = df['critical_temp']
x = df.drop(['critical_temp'], axis=1).drop(['Unnamed: 0'], axis = 1).drop(['Unnamed: 0.1'], axis = 1).select_dtypes(exclude=['object'])

#creating the testing and training amostrages
train_X, test_X, train_y, test_y = train_test_split(x.values, y.values, test_size=0.2)
df_imputer = SimpleImputer()
train_X = df_imputer.fit_transform(train_X)
test_X = df_imputer.transform(test_X)

#xgboost
model = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.50,learning_rate = 0.02, max_depth = 16, alpha = 1, n_estimators = 374)
model.fit(train_X, train_y, verbose=False) # fitting the model

avg = 0
predictions = 0
for i in range(1, 26):
    predictions = model.predict(test_X)
    avg += sqrt(mean_absolute_error(predictions, test_y))

print("XGBoost Regressor rmse value: " + str(avg/25))
mailMessage+= "\nXGBoost Regressor rmse value: " + str(avg/25)

#plotting the image
plt.title("Predict Tc versus Observed Tc using XGBoost Regressor")
plt.plot(test_y, predictions, "o", color="black")
plt.plot(range(-10, 125), range(-10, 125), color = 'gray')
plt.xlabel("Observed Tc(K)")
plt.ylabel("Predicted Tc(K)")
#plt.show -> plt.savefig
#plt.savefig(PLOT_IMAGE_NAME)
plt.show()

#testing the model with the test element
print("XGBoost Regressor predicted value for Y1Ba2Cu3O7: ", model.predict(ybaco7)[0])
mailMessage+= "\nXGBoost Regressor data predicted value for Y1Ba2Cu3O7: " + str(model.predict(ybaco7)[0])

#random forests
forestModel = RandomForestRegressor(n_estimators=60)
forestModel.fit(train_X, train_y)

avg_forest = 0
predictions_forest = 0
for i in range(1, 26):
    predictions_forest = forestModel.predict(test_X) 
    avg_forest += sqrt(mean_absolute_error(predictions_forest, test_y))

print('RandomForests Regressor RMSE(for comparison): ', str(avg_forest/25))
mailMessage += '\nRandomForests Regressor RMSE(for comparison):' + str(avg_forest/25)

plt.title("Predict Tc versus Observed Tc using RandomForests Regressor")
plt.plot(test_y, predictions_forest, "o", color="black")
plt.plot(range(-10, 125), range(-10, 125), color = 'gray')
plt.xlabel("Observed Tc(K)")
plt.ylabel("Predicted Tc(K)")
#plt.show -> plt.savefig
plt.show()

print("RandomForests Regressor predicted value for Y1Ba2Cu3O7: ", forestModel.predict(ybaco7)[0])
mailMessage+= '\nRandomForests Regressor predicted value for Y1Ba2Cu3O7: ' + str(forestModel.predict(ybaco7)[0])

#knn
knnModel = KNeighborsRegressor(n_jobs=-1, weights='distance')
knnModel.fit(train_X, train_y)

avg_knn = 0
predictions_knn = 0
for i in range(1, 26):
    predictions_knn = knnModel.predict(test_X)
    avg_knn += sqrt(mean_absolute_error(predictions_knn, test_y))

print('KNN Regressor RMSE(for comparison): ', str(avg_knn/25))
mailMessage+= ('\nKNN Regressor RMSE(for comparison): ' + str(avg_knn/25))

plt.title("Predict Tc versus Observed Tc using KNN Regressor")
plt.plot(test_y, predictions_knn, "o", color="black")
plt.plot(range(-10, 125), range(-10, 125), color = 'gray')
plt.xlabel("Observed Tc(K)")
plt.ylabel("Predicted Tc(K)")
#plt.show -> plt.savefig
plt.show()

print("KNN Regressor predicted value for Y1Ba2Cu3O7: ", knnModel.predict(ybaco7)[0])
mailMessage+= '\nKNN Regressor data: Predicted value for Y1Ba2Cu3O7: ' + str(knnModel.predict(ybaco7)[0])    
#sending mail
#mailSender(PLOT_IMAGE_NAME, mailMessage)