import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import split
from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.statespace.tools import diff
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.tsa.holtwinters as ets

def split_dataset(data):
    # split into standard weeks
    train, test = data[:-220], data[-220:-3]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))

# evaluate a single model
def evaluate_model(model_func, train, test):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = model_func(history)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    predictions = array(predictions)
    predictions2 = predictions.flatten()
    # evaluate predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, predictions2

# convert windows of weekly multivariate data into a series of total power
def to_series(data):
    # extract just the total power from each week
    series = [week[:, 0] for week in data]
    # flatten into a single series
    series = array(series).flatten()
    return series

# holt forecast
def holt_forecast(history):
    # convert history into a univariate series
    series = to_series(history)
    # define the model
    # model = ARIMA(series, order=(6,0,0))


    model = ets.ExponentialSmoothing(series, trend='add', damped=True, seasonal='add', seasonal_periods=7)

    # fit the model
    model_fit = model.fit()
    # make forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

# load file
df = read_csv('timeseriesdata.csv', index_col="Unnamed: 0", parse_dates=True)
series = df[['Count']]
# split into train and test
train, test = split_dataset(series.values)
# define the names and functions for the models we wish to evaluate
models = dict()
models['holt'] = holt_forecast
# evaluate each model
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
for name, func in models.items():
    # evaluate and get scores
    score, scores, pred = evaluate_model(func, train, test)
    # summarize scores
    summarize_scores(name, score, scores)
    # plot scores
    pyplot.plot(days, scores, marker='o', label=name)
    pyplot.title("Holt-Winter Model - RSME by Weekday")

# show plot
pyplot.show()



train_holt, test_holt = df[:-220], df[-220:-3]
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
test_holt['Predictions'] = pred

# Plot Predictions
fig, ax = plt.subplots()
ax.plot(train_holt.Count, label="Train Set")
ax.plot(test_holt['Count'], label="Test Set")
ax.plot(test_holt['Predictions'], label="Predicted Values")
ax.set_xlabel("Time(t)")
ax.set_ylabel("Daily Traffic Accident Count")
ax.set_title("Daily Traffic Accident Prediction - Holt-Winter Method")
ax.legend()
plt.show()

# Plot Predictions for Average Method
test_holt["Pred2"] = 283
fig, ax = plt.subplots()
ax.plot(train_holt.Count, label="Train Set")
ax.plot(test_holt['Count'], label="Test Set")
ax.plot(test_holt['Pred2'], label="Average")
ax.set_xlabel("Time(t)")
ax.set_ylabel("Daily Traffic Accident Count")
ax.set_title("Daily Traffic Accident Count Chicago, IL")
ax.legend()
plt.show()

# Plot Sample Weekly Forecast
pred2=np.round(pred[:7],2)

days = ['Sunday, April 26', 'Monday, April 27', 'Tuesday, April 28',
        'Wednesday, April 29', 'Thursday, April 30', 'Friday, April 30',
        'Saturday, May 1']
plt.plot(days, pred2, marker='o', c="c", lw=2)
plt.title("Daily Traffic Accident Prediction Week of Aoril 26")
plt.savefig('Holt.png')
for i,j in zip(days,pred2):
    plt.annotate(str(j),xy=(i,j))
plt.show()
