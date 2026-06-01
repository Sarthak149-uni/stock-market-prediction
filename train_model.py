import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout

data = yf.download("AAPL", start="2010-01-01")

data = data[['Close']]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(data)

x_train = []
y_train = []

for i in range(100,len(scaled)):
    x_train.append(scaled[i-100:i])
    y_train.append(scaled[i])

x_train = np.array(x_train)
y_train = np.array(y_train)

model = Sequential()

model.add(LSTM(50, return_sequences=True,
               input_shape=(100,1)))
model.add(Dropout(0.2))

model.add(LSTM(60, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(80))
model.add(Dropout(0.4))

model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=32
)

model.save("model/stock_lstm_model.keras")