import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("crop_yield.csv")

# Streamlit title
st.title("Crop Yield Prediction Using Random Forest")

# Display dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Preprocessing
X = df.drop(columns=['Yield']) 
y = df['Yield']
X = pd.get_dummies(X, columns=['Crop', 'Season', 'State'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Absolute Error: {mae}")
st.write(f"Mean Squared Error: {mse}")
st.write(f"R² Score: {r2}")

# Plot Actual vs Predicted Yield
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_xlabel("Actual Yield")
ax.set_ylabel("Predicted Yield")
ax.set_title("Actual vs Predicted Yield")
st.pyplot(fig)  # This replaces plt.show() in Streamlit
