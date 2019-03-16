import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset_train = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//LSTM//Train.csv')
training_set = dataset_train.iloc[:, 37:38].values
train = dataset_train.iloc[:, 37:38].values
training_set = training_set[~np.isnan(training_set)]
training_set= training_set.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(60, 4000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 40, batch_size = 32)



dataset_test = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//LSTM//Test.csv')
real_stock_price = dataset_test.iloc[:, 9:10].values


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

error = 0.0
for i in range(0,20):
    error += ((predicted_stock_price[i]-real_stock_price[i])/real_stock_price[i])*((predicted_stock_price[i]-real_stock_price[i])/real_stock_price[i])
    
error = math.sqrt(error)
print(error)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

err = real_stock_price[0] - predicted_stock_price[0]
for i in range(0,20):
    predicted_stock_price[i] += err
    
error = 0.0
for i in range(0,20):
    error += ((predicted_stock_price[i]-real_stock_price[i])/real_stock_price[i])*((predicted_stock_price[i]-real_stock_price[i])/real_stock_price[i])
    
error = math.sqrt(error)
print(error)

plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()





dataset_test = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//LSTM//Test2.csv')
real_stock_price = dataset_test.iloc[:, 37:38].values


dataset_total = pd.concat((dataset_train['KAMA'], dataset_test['KAMA']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 180):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



#plt.plot(training_set, color = 'black', label = 'Training Data')
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

predicted_stock_price_Train = regressor.predict(X_train)
predicted_stock_price_Train = sc.inverse_transform(predicted_stock_price_Train)

#plt.plot(training_set, color = 'black', label = 'Training Data')
plt.plot(train, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price_Train, color = 'blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



error = 0.0
for i in range(0,120):
    error += (predicted_stock_price[i]-real_stock_price[i])*(predicted_stock_price[i]-real_stock_price[i])
error/=120    
error = math.sqrt(error)
print(error)