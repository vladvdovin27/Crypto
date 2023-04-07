from tensorflow import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('BTC_data.csv')

data_size, current_index = 990, 0
data, target = [], []

# Data generate
for _ in range(data_size):
    h_lst = []
    for i in range(5):
        h_lst.append(df.loc[current_index + i]['Close'])
    scaler = StandardScaler()
    h_lst = scaler.fit_transform([h_lst]).reshape(5, 1)
    data.append(h_lst)
    target.append(np.array(df.loc[current_index + 5]['Close']))

data, target = np.array(data), np.array(target)

# Build model
model = keras.Sequential([
    keras.Input(shape=(5, 1)),
    layers.LSTM(16, activation='tanh', return_sequences=True),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.LSTM(32, activation='tanh', return_sequences=True),
    layers.BatchNormalization(),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.LSTM(64, activation='tanh', return_sequences=True),
    layers.BatchNormalization(),
    layers.LSTM(32, activation='tanh', return_sequences=True),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.LSTM(32, activation='tanh'),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
])

X_train, X_test, Y_train, Y_test = data[:900], data[900:], target[:900], target[900:]

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='mse', metrics=['mae'])

cb = [keras.callbacks.EarlyStopping(patience=5)]
history = model.fit(X_train, Y_train, epochs=75, batch_size=32, validation_split=0.15, callbacks=cb)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('LSTMModel.h5')
print(model.evaluate(X_test, Y_test))
