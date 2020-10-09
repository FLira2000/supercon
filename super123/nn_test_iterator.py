from sklearn.metrics import mean_absolute_error
from math import sqrt

def NNtest_iterator(nn, test_X, test_y, iter_times):
    avg = 0
    for i in range(1, iter_times):
        predictions = nn.predict(test_X)
        avg += sqrt(mean_absolute_error( predictions, test_y ))

    return predictions, avg/iter_times, nn