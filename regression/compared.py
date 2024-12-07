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
import time
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

X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
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

# Define initial hyperparameters for fairness
initial_hyperparams = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_split': 2,
    'max_features': None
}

# Train and evaluate the initial model
initial_model = RandomForestRegressor(
    n_estimators=initial_hyperparams['n_estimators'],
    max_depth=initial_hyperparams['max_depth'],
    min_samples_split=initial_hyperparams['min_samples_split'],
    max_features=initial_hyperparams['max_features'],
    random_state=42
)
initial_model.fit(X_train, y_train)
initial_r2 = r2_score(y_valid, initial_model.predict(X_valid))
print(f"Initial Model R² Score: {initial_r2:.4f}")

### EXP3 Algorithm
class EXP3:
    def __init__(self, n_arms, gamma=0.1):
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
    
    def select_arm(self):
        total_weight = np.sum(self.weights)
        probabilities = (
            (1 - self.gamma) * (self.weights / total_weight) +
            (self.gamma / self.n_arms)
        )
        arm = np.random.choice(self.n_arms, p=probabilities)
        return arm, probabilities
    
    def update(self, chosen_arm, reward, probabilities):
        x = reward / probabilities[chosen_arm]
        self.weights[chosen_arm] *= np.exp((self.gamma * x) / self.n_arms)

# EXP3 Tuning
exp3_params = initial_hyperparams.copy()
hyperparameter_values = {
    'n_estimators': list(range(50, 301, 10)),
    'max_depth': [None] + list(range(5, 101, 10)),
    'min_samples_split': list(range(2, 21, 2)),
    'max_features': ['sqrt', 'log2', None]
}
n_iterations = 20

# Initialize the EXP3 bandits
exp3_bandits = {hp: EXP3(len(values)) for hp, values in hyperparameter_values.items()}
best_r2_exp3 = initial_r2
best_params_exp3 = initial_hyperparams.copy()
exp3_r2_progress = []


exp3_start_time = time.time()

for iteration in range(n_iterations):
    # Select a hyperparameter to tune
    selected_hp = np.random.choice(list(hyperparameter_values.keys()))
    bandit = exp3_bandits[selected_hp]
    
    # Select a value for the chosen hyperparameter
    arm, probabilities = bandit.select_arm()
    chosen_value = hyperparameter_values[selected_hp][arm]
    
    # Update the parameter
    old_value = exp3_params[selected_hp]
    exp3_params[selected_hp] = chosen_value

    # Train the model with the updated parameters
    model = RandomForestRegressor(
        n_estimators=exp3_params['n_estimators'],
        max_depth=exp3_params['max_depth'],
        min_samples_split=exp3_params['min_samples_split'],
        max_features=exp3_params['max_features'],
        random_state=42
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_valid, model.predict(X_valid))
    exp3_r2_progress.append(r2)

    # Update bandit
    reward = r2
    bandit.update(arm, reward, probabilities)
    
    # Calculate regret
    regret = best_r2_exp3 - r2
    
    # Update best R² if applicable
    if r2 > best_r2_exp3:
        best_r2_exp3 = r2
        best_params_exp3 = exp3_params.copy()
    else:
        exp3_params[selected_hp] = old_value  # Revert if not better

exp3_time = time.time() - exp3_start_time
print(f"EXP3 Tuning Time: {exp3_time:.2f} seconds")
print(f"Best R² via EXP3: {best_r2_exp3:.4f}")
print(f"Best Parameters via EXP3: {best_params_exp3}")

### Bayesian Optimization
space = [
    Integer(50, 300, name='n_estimators'),
    Integer(5, 100, name='max_depth'),
    Integer(2, 50, name='min_samples_split'),
    Categorical(['sqrt', 'log2', None], name='max_features')
]

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
    r2 = r2_score(y_valid, model.predict(X_valid))
    return -r2

bayes_start_time = time.time()
result = gp_minimize(objective, space, n_calls=n_iterations, random_state=42)
bayes_time = time.time() - bayes_start_time

best_r2_bayes = -result.fun
best_params_bayes = {
    'n_estimators': result.x[0],
    'max_depth': result.x[1],
    'min_samples_split': result.x[2],
    'max_features': result.x[3]
}
print(f"Bayesian Optimization Time: {bayes_time:.2f} seconds")
print(f"Best R² via Bayesian Optimization: {best_r2_bayes:.4f}")
print(f"Best Parameters via Bayesian Optimization: {best_params_bayes}")

### Comparative Plots
# R² Progress
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iterations + 1), exp3_r2_progress, label="EXP3 R²")
plt.plot(range(1, n_iterations + 1), -np.array(result.func_vals), label="Bayesian R²")
plt.xlabel("Iterations")
plt.ylabel("R² Score")
plt.title("R² Progress Comparison")
plt.legend()
plt.grid()
plt.show()



# Time Comparison
plt.bar(["EXP3", "Bayesian Optimization"], [exp3_time, bayes_time])
plt.title("Tuning Time Comparison")
plt.ylabel("Time (seconds)")
plt.show()
