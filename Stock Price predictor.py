import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score

# Download historical stock data for Google
data = yf.download("GOOGLE", start="2020-01-01", end="2025-01-01")
data.head(10)

# Plotting the closing price
plt.figure(figsize=(16,8))
plt.title('GOOGL Close Price History')
plt.plot(data['Close'],label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.grid()
plt.show()

# Create features based on price differences
data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna()

# Define features (X) and target for classification (Y)
X = data[['Open - Close','High - Low']]
Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

# Split data for classification
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

# Find the best number of neighbors for KNN Classifier
params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

# Fit the classification model
model.fit(X_train, y_train)

# Check the accuracy of the classification model
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))
print("Train_data accuracy: %.2f" % accuracy_train)
print("Test_data accuracy: %.2f" % accuracy_test)

# Define target for regression (y)
y = data['Close']

# Split data for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state=44)

# Find the best number of neighbors for KNN Regressor
params_reg = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn_reg = KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params_reg, cv=5)

# Fit the regression model
model_reg.fit(X_train_reg, y_train_reg)

# Make predictions with the regression model
predictions = model_reg.predict(X_test_reg)

# Calculate the root mean squared error
rms = np.sqrt(np.mean(np.power((np.array(y_test_reg) - np.array(predictions)), 2)))
print("Root Mean Squared Error (RMSE):", rms)

# Hypothetical daily values:
today_open = 181.50
today_high = 184.00
today_low  = 180.75
# To create the 'Open - Close' feature, you need a 'Close' value.
# We'll use a hypothetical close to construct the feature.
hypothetical_close_for_feature = 183.25

# 1. Calculate the features in the correct format
feature_1 = today_open - hypothetical_close_for_feature  # 'Open - Close'
feature_2 = today_high - today_low                       # 'High - Low'
today_features = np.array([[feature_1, feature_2]])

# 2. Use the REGRESSION model ('model_reg') to predict a numerical price
predicted_price = model_reg.predict(today_features)
print(f"\nPredicted Close Price for Today (using Regressor): {predicted_price[0]:.2f}")

# 3. Use the CLASSIFICATION model ('model') to predict the direction for the NEXT day
predicted_direction = model.predict(today_features)
direction = "Up" if predicted_direction[0] == 1 else "Down"
print(f"Predicted Direction for Next Day (using Classifier): {direction}")