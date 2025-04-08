# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

if not os.path.exists('model'):
    os.makedirs('model')

# Read the dataset
df = pd.read_csv("data/rental_1000.csv")

# Feature engineering: Select Features (X) and Label (y)
X = df[['rooms', 'area']].values
y = df['price'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

# Train and evaluate Linear Regression
lr = LinearRegression()
lr_model = lr.fit(X_train, y_train)

predictions = lr_model.predict(X_test)
print("Actual prices vs predicted prices", y_test[:5], predictions[:5])

score = lr_model.score(X_test, y_test)
rmse = np.sqrt(np.mean((lr_model.predict(X_test)-y_test) ** 2))
print(f"Model score: {score}")
print(f"RMSE : {rmse}")

# Train and evaluate Random Forest
rf = RandomForestRegressor(random_state=42)
rf_model = rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
rf_rmse = np.sqrt(np.mean((rf.predict(X_test) - y_test) ** 2))
print(f"Random Forest - R^2 Score: {rf_score:.4f}, RMSE: {rf_rmse:.4f}")

# Train and evaluate K-Nearest Neighbors
knn = KNeighborsRegressor()
knn_model = knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)
knn_rmse = np.sqrt(np.mean((knn.predict(X_test) - y_test) ** 2))
print(f"KNN - R^2 Score: {knn_score:.4f}, RMSE: {knn_rmse:.4f}")



# Save the best model (Linear Regression in this case) to a file
model = lr_model
model_path = 'model/rental_prediction_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved successfully at {model_path}!")


def predict_price(rooms, area):
    with open('model/rental_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([[rooms, area]])
    return prediction[0]

print(predict_price(4, 120))
