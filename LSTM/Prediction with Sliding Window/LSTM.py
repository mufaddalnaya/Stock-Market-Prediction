import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


dataset_train = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//TestTrain//Train.csv')
training_set = dataset_train.iloc[:, 9:10].values
train = dataset_train.iloc[:, 9:10].values
training_set = training_set[~np.isnan(training_set)]
training_set= training_set.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


regressor = Sequential()
regressor.add(LSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 45, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 30, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 15))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1)) 

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 40, batch_size = 60)

#predicting training data
training = []
for i in range(60,len(training_set)):
    training.append(training_set[i])
training = np.array(training)
predicted_stock_priceTrain = regressor.predict(X_train)
predicted_stock_priceTrain = sc.inverse_transform(predicted_stock_priceTrain)

rms = np.sqrt(np.mean(np.power((training-predicted_stock_priceTrain)/training,2)))
rms

   
# Visualising the results
plt.plot(training, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_priceTrain, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



#Predicting testing data
sc1 = MinMaxScaler(feature_range = (0, 1))
dataset_test = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//TestTrain//Test.csv')
real_stock_price = dataset_test.iloc[:, 9:10].values


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs[~np.isnan(inputs)]
inputs = inputs.reshape(-1,1)
inputs = sc1.fit_transform(inputs)
X_test = []
Y_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc1.inverse_transform(predicted_stock_price)




rms = np.sqrt(np.mean(np.power((real_stock_price-predicted_stock_price)/real_stock_price,2)))
rms


# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
