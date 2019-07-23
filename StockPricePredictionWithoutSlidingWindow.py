# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET 
dataset = pd.read_csv('C://Users//abc//Downloads//Stock-Price-Prediction-LSTM-master//NEWDATANFLX.csv', usecols=[1,2,3,4,5,9])
dataset = dataset.reindex(index = dataset.index[::-1])

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

# PLOTTING ALL INDICATORS IN ONE PLOT
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
plt.plot(obs, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'upper right')
plt.show()


close_val = dataset[['MA for 14 days']]
close_val = np.reshape(close_val.values, (len(close_val),1))

train_OHLC = int(len(close_val) * 0.75)
test_OHLC = len(close_val) - train_OHLC
train_OHLC, test_OHLC = close_val[0:train_OHLC,:], close_val[train_OHLC:len(close_val),:]

scaler1 = MinMaxScaler(feature_range=(0, 1))
train_OHLC = scaler1.fit_transform(train_OHLC)

scaler2 = MinMaxScaler(feature_range=(0, 1))
test_OHLC = scaler2.fit_transform(test_OHLC)


# PREPARATION OF TIME SERIES DATASE
close_val = np.reshape(close_val.values, (len(close_val),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
close_val = scaler.fit_transform(close_val)

# TRAIN-TEST SPLIT
train_OHLC = int(len(close_val) * 0.95)
test_OHLC = len(close_val) - train_OHLC
train_OHLC, test_OHLC = close_val[0:train_OHLC,:], close_val[train_OHLC:len(close_val),:]

def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)


# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL
model = Sequential()
model.add(LSTM(60, input_shape=(1, step_size), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 45, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 30, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 15))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
#model.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
#model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
model.fit(trainX, trainY, epochs=40, batch_size=60)
# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler1.inverse_transform(trainPredict)
trainY = scaler1.inverse_transform([trainY])
testPredict = scaler2.inverse_transform(testPredict)
testY = scaler2.inverse_transform([testY])

# TRAINING RMSE
trainScore = np.sqrt(np.mean(np.power(trainY[0]-trainPredict[:,0],2)))
print('Train RMSE: %f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(close_val)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(close_val)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(close_val)-1, :] = testPredict

# DE-NORMALIZING MAIN DATASET 
close_val = scaler.inverse_transform(close_val)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(close_val, 'g')
plt.plot(trainPredictPlot, 'r')
plt.plot(testPredictPlot, 'b')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('SMA for 14 days Value of Netflix Stocks')
plt.show()


plt.plot(trainY[0], 'g', label = 'original Training dataset')
plt.plot(trainPredict, 'b', label = 'Predicted Stock Price')
plt.legend()
plt.xlabel('Time in Days')
plt.ylabel('SMA for 14 days Value of Netflix Stocks')
plt.show()


plt.plot(testY[0], 'g', label = 'original Testing dataset')
plt.plot(testPredict, 'b',label = 'Predicted Stock Price')
plt.xlabel('Time in Days')
plt.ylabel('SMA for 14 days Value of Netflix Stocks')
plt.legend()
plt.show()


plt.plot(close_val, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')













# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print "Last Day Value:", np.asscalar(last_val)
print "Next Day Value:", np.asscalar(last_val*next_val)
# print np.append(last_val, next_val)































































































