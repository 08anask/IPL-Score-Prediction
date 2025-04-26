# ----------------------------------------------
# Import Required Libraries
# ----------------------------------------------
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
import joblib

# ----------------------------------------------
# Load the Dataset
# ----------------------------------------------
df = pd.read_csv("data//cleaned.csv")

# ----------------------------------------------
# One-Hot Encode Categorical Features
# ----------------------------------------------
# Convert categorical features 'bat_team' and 'bowl_team' into one-hot encoded columns
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# ----------------------------------------------
# Select Relevant Columns
# ----------------------------------------------
# Keep only the necessary columns for modeling
encoded_df = encoded_df[[
    'date', 
    'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
    'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
    'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
    'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
    'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
    'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
    'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total'
]]

# ----------------------------------------------
# Split Data into Training and Test Sets
# ----------------------------------------------
# Convert 'date' column to datetime
encoded_df['date'] = pd.to_datetime(encoded_df['date'])

# Separate features and target variable based on year
X_train = encoded_df.drop('total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop('total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Remove 'date' from features
X_train.drop('date', axis=1, inplace=True)
X_test.drop('date', axis=1, inplace=True)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# ----------------------------------------------
# Linear Regression Model
# ----------------------------------------------
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_lr = linear_regressor.predict(X_test)

print("\n---- Linear Regression - Model Evaluation ----")
print("MAE:", mae(y_test, y_pred_lr))
print("MSE:", mse(y_test, y_pred_lr))
print("RMSE:", np.sqrt(mse(y_test, y_pred_lr)))

# ----------------------------------------------
# Decision Tree Regressor
# ----------------------------------------------
decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train, y_train)
y_pred_dt = decision_regressor.predict(X_test)

print("\n---- Decision Tree Regression - Model Evaluation ----")
print("MAE:", mae(y_test, y_pred_dt))
print("MSE:", mse(y_test, y_pred_dt))
print("RMSE:", np.sqrt(mse(y_test, y_pred_dt)))

# ----------------------------------------------
# Random Forest Regressor
# ----------------------------------------------
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train, y_train)
y_pred_rf = random_regressor.predict(X_test)

print("\n---- Random Forest Regression - Model Evaluation ----")
print("MAE:", mae(y_test, y_pred_rf))
print("MSE:", mse(y_test, y_pred_rf))
print("RMSE:", np.sqrt(mse(y_test, y_pred_rf)))

# ----------------------------------------------
# AdaBoost Regressor (using Linear Regression as base)
# ----------------------------------------------
adb_regressor = AdaBoostRegressor(base_estimator=linear_regressor, n_estimators=100)
adb_regressor.fit(X_train, y_train)
y_pred_adb = adb_regressor.predict(X_test)

print("\n---- AdaBoost Regression - Model Evaluation ----")
print("MAE:", mae(y_test, y_pred_adb))
print("MSE:", mse(y_test, y_pred_adb))
print("RMSE:", np.sqrt(mse(y_test, y_pred_adb)))


# ----------------------------------------------
# Save the Best Performing Model - Linear Regression
# ----------------------------------------------
joblib.dump(linear_regressor, 'linear_regression_model.pkl')
print("\nLinear Regression model saved as 'linear_regression_model.pkl'")