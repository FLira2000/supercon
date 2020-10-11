import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
import os
from sklearn.model_selection import train_test_split
from neuralNets.NN_logisticActiv_doubleLayer import neural_logistic_2layer
from neuralNets.NN_logisticActiv_singleLayer import neural_logistic_1layer
from neuralNets.NN_reluActiv_doubleLayer import neural_relu_2layer
from neuralNets.NN_reluActiv_singleLayer import neural_relu_1layer
from neuralNets.NN_tahnActiv_doubleLayer import neural_tahn_2layer
from neuralNets.NN_tahnActiv_singleLayer import neural_tahn_1layer
from nn_test_iterator import NNtest_iterator

df = pd.read_csv(f"./databases/super123Database_final.csv")
df = df.drop("Unnamed: 0", axis=1).drop("Unnamed: 0.1", axis=1)

y = df['critical_temp']

x = df.select_dtypes(exclude=['object']).drop('critical_temp', axis=1)

train_X, test_X, train_y, test_y = train_test_split(x.values, y.values, test_size=0.2)
df_imputer = SimpleImputer()
train_X = df_imputer.fit_transform(train_X)
test_X = df_imputer.transform(test_X)

predictions, score, nn_model = NNtest_iterator(neural_relu_1layer(), test_X, test_y, 30)

print(score)


