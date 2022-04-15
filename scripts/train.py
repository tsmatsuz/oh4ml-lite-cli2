import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import argparse

from pandas import Grouper
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

# parse argument
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_file',
    type=str)
parser.add_argument(
    '--model_folder',
    type=str,
    default='./outputs',)
FLAGS, unparsed = parser.parse_known_args()

df = pd.read_csv(FLAGS.data_file)

df.set_index('TransactionDate',inplace=True)
df.columns = ['PaidAmount']
series = pd.Series(df['PaidAmount'])

def mean_and_variance(X):
    split = int(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('mean1=%f, mean2=%f' % (mean1, mean2))
    print('variance1=%f, variance2=%f' % (var1, var2))

mean_and_variance(series.values)

def fuller_test(X):
    result = adfuller(X)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
    	print('\t%s: %.3f' % (key, value))

fuller_test(series.values)

plot_acf(series)

plot_pacf(series)

X = series.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]

model = ARIMA(train, order=(2,0,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(title="Residuals Error Plot")
plt.show()
#residuals.plot(kind='kde')
#plt.show()
#print(residuals.describe())

predictions=model_fit.forecast(steps=test.size)[0]

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test,predictions)
print('Test RMSE: %.3f' % rmse)
print('Test R2: %.3f' % r2)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.title("Test Data Vs. Predictions")
plt.show()

model_file_name = 'arima_model.pkl'

os.makedirs(FLAGS.model_folder, exist_ok=True)
with open(model_file_name, 'wb') as file:
    model_path = os.path.join(FLAGS.model_folder, model_file_name)
    joblib.dump(
        value=model_fit,
        filename=model_path)

print(
    "Model saved: {}".format(
        model_path
    )
)