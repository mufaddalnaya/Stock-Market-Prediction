import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset_train = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//Train.csv')
training_set = dataset_train.iloc[:, 37:38].values
train = dataset_train.iloc[:, 37:38].values
training_set = training_set[~np.isnan(training_set)]
training_set= training_set.reshape(-1, 1)

X_train = []
y_train = []
for i in range(60, 4000):
    X_train.append(training_set[i-60:i, 0])
    y_train.append(training_set[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

model = Sequential()

# Adding the first CNN layer and some Dropout regularisation
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))

model.add(MaxPooling1D(pool_size=2))
# Adding a second LSTM layer and some Dropout regularisation

model.add(Flatten())
# Adding a third LSTM layer and some Dropout regularisation

model.add(Dense(50, activation='relu'))

# Adding the output layer
model.add(Dense(1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 40, batch_size = 32)



dataset_test = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//Test.csv')
real_stock_price = dataset_test.iloc[:, 9:10].values


dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
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





dataset_test = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//Test2.csv')
real_stock_price = dataset_test.iloc[:, 37:38].values


dataset_total = pd.concat((dataset_train['KAMA'], dataset_test['KAMA']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
X_test = []
for i in range(60, 180):
    X_test.append(inputs[i-60:i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)



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