import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('BTC_data.csv')
to_display_df = pd.read_csv('BTC_data.csv')
model = load_model('LSTMModel.h5')

current_data = np.array(df.loc[995:]['Close']).reshape(1, 5, 1)

for i in range(12):
    print(current_data)
    scaler = StandardScaler()
    current_data = scaler.fit_transform(current_data.reshape(1, 5)).reshape(5, 1)
    predict = model.predict(current_data)
    print(predict)
    df.loc[len(df.index)] = predict[0][0]
    current_data = np.array(df.loc[995 + i + 1:]['Close']).reshape(1, 5, 1)

df['Close'].plot(legend=True)
to_display_df['Close'].plot(legend=True)
plt.show()
