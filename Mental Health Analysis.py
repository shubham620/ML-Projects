import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Create the demo dataset
data = {
    'Age': [20, 22, 19, 25, 23, 21, 24, 20, 22, 25],
    'Sleep_Hours': [6, 8, 7, 5, 7, 9, 6, 8, 7, 5],
    'Stress_Level': ['High', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High'],
    'Social_Support': ['Low', 'High', 'Medium', 'Low', 'Medium', 'High', 'Medium', 'High', 'Medium', 'Low'],
    'Academic_Stress': ['High', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High'],
    'Depression_Risk': ['High', 'Low', 'Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High']
}
df = pd.DataFrame(data)

# Define mappings to convert categorical string data to numerical data
level_map = {'Low': 0, 'Medium': 1, 'High': 2}
risk_map = {'Low': 0, 'Medium': 1, 'High': 2}

# Apply the mappings
df['Stress_Level'] = df['Stress_Level'].map(level_map)
df['Social_Support'] = df['Social_Support'].map(level_map)
df['Academic_Stress'] = df['Academic_Stress'].map(level_map)
df['Depression_Risk'] = df['Depression_Risk'].map(risk_map)

# Define the features (X) and the target variable (y)
X = df[['Age', 'Sleep_Hours', 'Stress_Level', 'Social_Support', 'Academic_Stress']]
y = df['Depression_Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("-" * 25)

# Get the importance of each feature from the trained model
feat_importance = model.feature_importances_
feat_names = X.columns

# Create a horizontal bar plot to visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feat_names, feat_importance)
plt.xlabel("Feature Importance")
plt.title("Which Features Matter Most?")
plt.gca().invert_yaxis() # Display the most important feature at the top
plt.show()

# Define a new data point for prediction
special_person = pd.DataFrame([{
    'Age': 20,
    'Sleep_Hours': 5,
    'Stress_Level': 2,       # Corresponds to 'High'
    'Social_Support': 0,     # Corresponds to 'Low'
    'Academic_Stress': 1     # Corresponds to 'Medium'
}])

# Use the trained model to predict the risk
predicted_risk_encoded = model.predict(special_person)[0]

# Create a reverse map to decode the numerical prediction back to a string
risk_map_rev = {0: 'Low', 1: 'Medium', 2: 'High'}
predicted_risk_decoded = risk_map_rev[predicted_risk_encoded]

# Print the final prediction for the new data point
print("\n--- New Prediction ---")
print(f"Predicted Depression Risk: {predicted_risk_decoded}")
print("-" * 25)
