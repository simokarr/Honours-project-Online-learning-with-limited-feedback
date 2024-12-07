# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Load the House Prices dataset
data = pd.read_csv('train2.csv')

# Preprocessing steps
# Drop columns with too many missing values or irrelevant features
data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], axis=1)

# Separate features and target
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Handle missing values
numeric_features = X.select_dtypes(include=[np.number]).columns
categorical_features = X.select_dtypes(include=[object]).columns

# Impute numerical features with median
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())

# Impute categorical features with mode
X[categorical_features] = X[categorical_features].fillna(X[categorical_features].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate an initial model
initial_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=2,
    max_features=None,
    random_state=42
)
initial_model.fit(X_train, y_train)
initial_y_pred = initial_model.predict(X_valid)
initial_r2 = r2_score(y_valid, initial_y_pred)
print(f"Initial Model R² Score: {initial_r2:.4f}")

# Define search space for Bayesian Optimization
space = [
    Integer(50, 300, name='n_estimators'),
    Integer(5, 50, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Categorical(['sqrt', 'log2', None], name='max_features')
]

# Define the objective function for Bayesian Optimization
@use_named_args(space)
def objective(**params):
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        max_features=params['max_features'],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    r2 = r2_score(y_valid, y_pred)
    return -r2  # Minimize negative R² score

# Run Bayesian Optimization
n_calls = 50  # Number of iterations for tuning
result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

# Retrieve the best parameters and accuracy
best_params_bo = {
    'n_estimators': result.x[0],
    'max_depth': result.x[1],
    'min_samples_split': result.x[2],
    'max_features': result.x[3]
}
print("\nBest Hyperparameters:")
print(best_params_bo)

# Train and evaluate final model with best parameters
final_model_bo = RandomForestRegressor(
    n_estimators=best_params_bo['n_estimators'],
    max_depth=best_params_bo['max_depth'],
    min_samples_split=best_params_bo['min_samples_split'],
    max_features=best_params_bo['max_features'],
    random_state=42
)
final_model_bo.fit(X_train, y_train)
final_y_pred_bo = final_model_bo.predict(X_valid)
final_r2_bo = r2_score(y_valid, final_y_pred_bo)
print(f"Final R² Score (Bayesian Optimization): {final_r2_bo:.4f}")

# Cumulative Regret Calculation
cumulative_regret_bo = []
best_possible_reward = max(initial_r2, final_r2_bo)
for i in range(1, len(result.func_vals) + 1):
    regret = best_possible_reward - (-result.func_vals[:i].min())
    if cumulative_regret_bo:
        cumulative_regret_bo.append(cumulative_regret_bo[-1] + regret)
    else:
        cumulative_regret_bo.append(regret)

# Plot cumulative regret
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_regret_bo) + 1), cumulative_regret_bo, label="Cumulative Regret")
plt.xlabel("Iterations")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret for Bayesian Optimization")
plt.legend()
plt.grid()
plt.show()

# Normalized Regret (R_T / T)
normalized_regret = [reg / t for reg, t in zip(cumulative_regret_bo, range(1, len(cumulative_regret_bo) + 1))]
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(normalized_regret) + 1), normalized_regret, label="Normalized Regret (R_T / T)")
plt.xlabel("Iterations")
plt.ylabel("Normalized Regret")
plt.title("Normalized Regret for Bayesian Optimization")
plt.legend()
plt.grid()
plt.show()
