import requests
import json
import pandas as pd

# Variables for make get response
currency = 'BTSUSDT'
interval = '5m'
root_url = 'https://api.binance.com/api/v3/klines'

# Get a data for train model
url = root_url + '?symbol=' + currency + '&interval=' + interval + '&limit=1000'
data = json.loads(requests.get(url).text)
df = pd.DataFrame(data)

# Print information about dataFrame
print(df.info())

data = pd.DataFrame()
data['Open'] = df[1]
data['High'] = df[2]
data['Low'] = df[3]
data['Close'] = df[4]

data.to_csv('BTC_data.csv')
