print("Note: This computation may take a while to be performed. Please wait. ")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from thermo import simple_formula_parser
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.application import MIMEApplication
from email import encoders
from os.path import basename
import datetime

agora = datetime.datetime.now()
mailMessage = str(agora.strftime("%Y-%m-%d %H.%M")) + "\n \n"
mailConfig = { "email": "data.supercon@gmail.com", "password": "rnzizqregtssljji" }
PLOT_IMAGE_NAME = "plot_" + agora.strftime("%Y-%m-%d_%H.%M") + ".png"

#importing the dataframe
df = pd.read_csv('ybaco_materials_all.csv')
df.head()

y = df['critical_temp']
x = df.drop(['critical_temp'], axis=1).drop(['Unnamed: 0'], axis = 1).drop(['Unnamed: 0.1'], axis = 1).select_dtypes(exclude=['object'])

#creating the testing and training amostrages
train_X, test_X, train_y, test_y = train_test_split(x.values, y.values, test_size=0.2)

df_imputer = SimpleImputer()
train_X = df_imputer.fit_transform(train_X)
test_X = df_imputer.transform(test_X)


#xgboost
#WARNING: this computation takes a while.

avg = 0
predictions = 0
model = XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.50,learning_rate = 0.02, max_depth = 16, alpha = 1, n_estimators = 374)
model.fit(train_X, train_y, verbose=False) # treinando o modelo

for i in range(1, 26):
    predictions = model.predict(test_X) # testando a predição
    avg += mean_absolute_error(predictions, test_y)

print("rmse value: " + str(sqrt(avg/25)))
mailMessage+= "rmse value: " + str(sqrt(avg/25))

#plotting the image
plt.title("Predict Tc versus Observed Tc")
plt.plot(test_y, predictions, "o", color="black")
plt.plot(range(-10, 125), range(-10, 125), color = 'gray')
plt.xlabel("Observed Tc(K)")
plt.ylabel("Predicted Tc(K)")
#plt.show -> plt.savefig
plt.savefig(PLOT_IMAGE_NAME)

#testing the model with an element 
ybaco7 = df.loc[df['Unnamed: 0'] == 29]
ybaco7 = ybaco7.drop(['material'], axis=1)
observedValue = np.array(ybaco7['critical_temp'])
ybaco7 = ybaco7.drop(['critical_temp'], axis=1)

ybaco7 = ybaco7.drop(['Unnamed: 0'], axis = 1).drop(['Unnamed: 0.1'], axis = 1).select_dtypes(exclude=['object'])

ybaco7 = np.array(ybaco7)

print("Testing predition with YBa2Cu3O7...")
print("Predicted value for Y1Ba2Cu3O7: ", model.predict(ybaco7)[0])
print("Observed value for Y1Ba2Cu3O7: ", observedValue[0])

mailMessage+= "\nPredicted value for Y1Ba2Cu3O7: " + str(model.predict(ybaco7)[0])
mailMessage+= "\nObserved value for Y1Ba2Cu3O7: " + str(observedValue[0])

#sending results over email
'''targetEmail = "fabioliradev@gmail.com"
ccEmail = "josiasdsj1@gmail.com"

msg = MIMEMultipart()
msg['From'] = "data.supercon@gmail.com"
msg['To'] = targetEmail
msg['Subject'] = "Prediction data"
msg['Cc'] = ', '.join([ccEmail])

body = mailMessage

msg.attach(MIMEText(body, 'plain'))

part = MIMEApplication(open(PLOT_IMAGE_NAME, "rb").read(), Name=basename(PLOT_IMAGE_NAME))
part['Content-Disposition'] = 'attachment; filename="%s"' % basename(PLOT_IMAGE_NAME)

msg.attach(part)

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(mailConfig["email"], mailConfig["password"])
s.send_message(msg)
s.quit()'''

#chemtest
ybaco = df['material'][29]
ybacoFormula = simple_formula_parser(ybaco)

elementList = []
for chem in ybacoFormula:
    elementList.append(chem)

elementList = np.array(elementList)
elementList = elementList[np.newaxis, :]
elementList = np.resize(elementList, (2, 4))

shadowList = []
for chem in ybacoFormula.values():
    shadowList.append(chem)

shadowList = np.array(shadowList)

elementList[1] = shadowList
print(elementList)
#now elementList is our material, converted into a matrix.