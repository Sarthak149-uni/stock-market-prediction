import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model

model = load_model(
    "model/stock_lstm_model.keras"
)

def predict_stock(data):

    scaler = MinMaxScaler()

    close = data[['Close']]

    scaled = scaler.fit_transform(close)

    x_test = []

    for i in range(100,len(scaled)):
        x_test.append(
            scaled[i-100:i]
        )

    x_test = np.array(x_test)

    predictions = model.predict(
        x_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions
    )

    return predictions