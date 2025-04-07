import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data/rental_1000.csv")

X = df[['rooms', 'area']].values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

lr = LinearRegression()
model = lr.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Actual prices vs predicted prices", y_test[:5], predictions[:5])

score = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((model.predict(X_test)-y_test) ** 2))
print(f"Model score: {score}")
print(f"RMSE : {rmse}")

#Save the model

with open('model/Rental_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)


def predict_price(rooms, area):
    with open('model/Rental_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([[rooms, area]])
    return prediction[0]

print(predict_price(4, 120))