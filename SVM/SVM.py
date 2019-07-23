import pandas as pd

import numpy as np
#used for scientific calculation
from sklearn.svm import SVR
#used for ml
import matplotlib.pyplot as matplt
#used for plotting graph

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

dataset_train = pd.read_csv('C://Users//abc//Projects//Stock-Market-Prediction//Train.csv')
price = dataset_train.iloc[:, 9:10].values
price = price[~np.isnan(price)]
price= price.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(price)
train_size = int(len(dataset) * .99)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))

look_back = 20
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)


trainX, trainY = create_dataset(train, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# reshape input to be [samples, time steps, features]
trainY = trainY.reshape(len(trainY), 1)

svr_rbf = SVR(kernel='linear', C=1e3, gamma=0.002)
model = svr_rbf.fit(trainX,trainY.ravel())
model.get_params()
trainPredict = model.predict(trainX)



svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)

svr_lin.fit(y_train,X_train)
svr_poly.fit(date,price)
svr_rbf.fit(y_train,X_train)


#PLOT THE DATA ON THE GRAPH

matplt.scatter(X_train,y_train,color='black',label='data')
matplt.plot(svr_lin.predict(y_train),color='blue',label='Linear SVR')
matplt.plot(X_train,color='grey',label='Price')
#matplt.plot(date,svr_poly.predict(date),color='red',label='Polynomial SVR')
matplt.plot(svr_rbf.predict(y_train),color='green',label='RBF SVR')
matplt.xlabel('Dates')
matplt.ylabel('Price')
matplt.title('Support Vector Regression')
matplt.legend()
matplt.show()
