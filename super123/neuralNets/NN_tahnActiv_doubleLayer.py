import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from math import sqrt

def neural_tahn_2layer():
    df = pd.read_csv('../super123Database_final')
    df = df.drop('Unnamed: 0', axis=1).drop('Unnamed: 0.1', axis=1)

    y = df['critical_temp']
    x = df.select_dtypes(exclude=['object']).drop('critical_temp', axis=1)

    train_X, test_X, train_y, test_y = train_test_split(x.values, y.values, test_size=0.2)
    df_imputer = SimpleImputer()
    train_X = df_imputer.fit_transform(train_X)
    test_X = df_imputer.transform(test_X)

    neuralNet_tahn2 = MLPRegressor(hidden_layer_sizes = (40, 50), activation = 'tanh', max_iter=2000)
    neuralNet_tahn2.fit(train_X, train_y)

    return neuralNet_tahn2