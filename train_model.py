import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('dataset/city_day.csv')

df = df[['PM2.5','PM10','NO2','CO','SO2','AQI']]

df = df.dropna()

X = df[['PM2.5','PM10','NO2','CO']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model/model.pkl','wb'))

print("Model Trained Successfully")