# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the EXP3 algorithm class
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

# Load the Titanic dataset
data = pd.read_csv('train.csv')

# Preprocessing steps
# Drop unnecessary columns
data = data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Drop rows with missing values
data = data.dropna()

# Encode categorical variables
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print()

# Define the possible values for each hyperparameter with expanded search space
hyperparameter_values = {
    'n_estimators': list(range(50, 301, 5)),  # [50, 55, 60, ..., 300]
    'max_depth': [None] + list(range(5, 100, 5)),  # [None, 5, 10, ..., 95]
    'min_samples_split': list(range(2, 50, 2)),  # [2, 4, 6, ..., 48]
    'max_features': ['sqrt', 'log2', None]
}

# Initialize current hyperparameter values (starting point)
current_params = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'max_features': 'sqrt'
}

# **New Section: Train and Evaluate the Initial Model**
# Train the Random Forest model with the initial hyperparameters
initial_model = RandomForestClassifier(
    n_estimators=current_params['n_estimators'],
    max_depth=current_params['max_depth'],
    min_samples_split=current_params['min_samples_split'],
    max_features=current_params['max_features'],
    random_state=42
)
initial_model.fit(X_train, y_train)

# Evaluate the initial model on the validation set
initial_y_pred = initial_model.predict(X_valid)
initial_accuracy = accuracy_score(y_valid, initial_y_pred)
print(f"Initial Model Accuracy (Before Tuning): {initial_accuracy:.4f}")


#plotting the graph
cumulative_regret = []
best_possible_reward = initial_accuracy 


# Number of iterations for the hyperparameter tuning loop
n_iterations = 20

# Initialize the bandit algorithms for each hyperparameter
bandits = {}
for hp in hyperparameter_values:
    n_arms = len(hyperparameter_values[hp])
    bandits[hp] = {
        'algorithm': EXP3(n_arms)
    }

# Initialize the bandit algorithm for hyperparameter selection
hyperparameters = list(hyperparameter_values.keys())
n_hyperparameters = len(hyperparameters)
hyperparameter_bandit = EXP3(n_hyperparameters)

# Map hyperparameter indices to names
hyperparameter_indices = {idx: hp for idx, hp in enumerate(hyperparameters)}

# Track the best model performance and parameters
best_accuracy = initial_accuracy  # Start with the initial accuracy
best_params = current_params.copy()

# Hyperparameter tuning loop
for i in range(n_iterations):
    print(f"\nIteration {i+1}/{n_iterations}")
    
    # Select which hyperparameter to adjust using the hyperparameter bandit
    hp_arm_index, hp_probabilities = hyperparameter_bandit.select_arm()
    hp_to_adjust = hyperparameter_indices[hp_arm_index]
    
    # Completely random selection
    """ hp_to_adjust = np.random.choice(hyperparameters)
    hp_arm_index = hyperparameters.index(hp_to_adjust)

    # Simulate uniform probabilities for compatibility
    n_hyperparameters = len(hyperparameters)
    hp_probabilities = np.ones(n_hyperparameters) / n_hyperparameters
     """
    
    
    # Select the value for the chosen hyperparameter using its bandit algorithm
    algorithm = bandits[hp_to_adjust]['algorithm']
    arm_index, probabilities = algorithm.select_arm()
    hp_value = hyperparameter_values[hp_to_adjust][arm_index]
    
    # Save the previous value to revert if necessary
    previous_value = current_params[hp_to_adjust]
    
    # Update the current parameters with the new value
    current_params[hp_to_adjust] = hp_value
    
    # Train the Random Forest model with the updated hyperparameters
    model = RandomForestClassifier(
        n_estimators=current_params['n_estimators'],
        max_depth=current_params['max_depth'],
        min_samples_split=current_params['min_samples_split'],
        max_features=current_params['max_features'],
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    reward = accuracy  # Use accuracy as the reward


    #track best possible reward for plotting
    best_possible_reward = max(best_possible_reward, reward)


    # Calculate regret for this iteration
    iteration_regret = best_possible_reward - reward


    # Update cumulative regret
    if cumulative_regret:
        cumulative_regret.append(cumulative_regret[-1] + iteration_regret)
    else:
        cumulative_regret.append(iteration_regret)

    comparison_df = pd.DataFrame({'Actual': y_valid.values, 'Predicted': y_pred})
    print(f"Predictions vs. Actual Values for Iteration {i+1}:")
    print(comparison_df.head(200))  # Display the first few rows
    
    # Update the bandit algorithm for the chosen hyperparameter value
    algorithm.update(arm_index, reward, probabilities)
    
    # Update the hyperparameter selection bandit
    hyperparameter_bandit.update(hp_arm_index, reward, hp_probabilities)
    
    # Print detailed output
    print(f"Adjusted Hyperparameter: {hp_to_adjust}")
    print(f"Selection Probabilities for Hyperparameters:")
    for idx, hp in enumerate(hyperparameters):
        print(f"  Hyperparameter: {hp}, Probability: {hp_probabilities[idx]:.4f}")
    print(f"Selection Probabilities for '{hp_to_adjust}':")
    for idx, val in enumerate(hyperparameter_values[hp_to_adjust]):
        print(f"  Value: {val}, Probability: {probabilities[idx]:.4f}")
    print(f"Chosen Value for '{hp_to_adjust}': {hp_value}")
    print(f"Reward (Accuracy): {accuracy:.4f}")
    print(f"Updated Weights for '{hp_to_adjust}': {algorithm.weights}")
    
    # Check if the new configuration is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = current_params.copy()
        print(f"New Best Accuracy: {best_accuracy:.4f} with Parameters: {best_params}")
    else:
        # If not better, revert the hyperparameter change
        current_params[hp_to_adjust] = previous_value

# After the tuning loop, print the best hyperparameters and accuracy
print("\nBest Hyperparameters Found:")
print(best_params)
print(f"Best Validation Accuracy: {best_accuracy:.4f}")

# Train the final model with the best hyperparameters
final_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    max_features=best_params['max_features'],
    random_state=42
)
final_model.fit(X_train, y_train)

# Make predictions on the validation set
final_predictions = final_model.predict(X_valid)

# Compare final predictions with actual values
final_comparison_df = pd.DataFrame({'Actual': y_valid.values, 'Predicted': final_predictions})
print("\nFinal Predictions vs. Actual Values:")
print(final_comparison_df.head(200))  # Display the first few rows


# Plot cumulative regret and compare it with sublinear growth (k * T * log(T))
k = n_arms  # You can adjust this constant as needed

T = np.arange(1, n_iterations+1)  # Example range of rounds

# Calculate the values for sqrt(K * T * log(K))
log_fit = np.sqrt(k* T * np.log(k))


plt.figure(figsize=(10, 6))

# Plot cumulative regret
plt.plot(range(0, n_iterations), cumulative_regret, label="Cumulative Regret")
plt.plot(range(0, n_iterations ), log_fit, linestyle='--', label=r"$\sqrt{KT \log(K)}$ Sublinear Fit")

# Add labels and legend
plt.xlabel("Iterations")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret of EXP3 (Sublinear Comparison)")
plt.legend()
plt.grid()  
plt.show()

# Plot normalized regret (R_T / T) to show that it decreases over iterations
normalized_regret = [reg / t for reg, t in zip(cumulative_regret, range(1, n_iterations + 1))]

plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iterations + 1), normalized_regret, label="Normalized Regret (R_T / T)")
plt.xlabel("Iterations")
plt.ylabel("R_T / T")
plt.title("Normalized Regret (R_T / T) vs. Iterations")
plt.legend()
plt.grid()
plt.show()



""" 

# Load the test.csv file
test_data = pd.read_csv('test.csv')

# Preprocessing for the test data
# Drop unnecessary columns
test_data = test_data.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, errors='ignore')

# Fill or drop missing values (depending on your training preprocessing)
test_data = test_data.dropna()

# Encode categorical variables
for column in ['Sex', 'Embarked']:
    if column in test_data.columns:
        if column in label_encoders:  # Use the same LabelEncoders from training
            test_data[column] = label_encoders[column].transform(test_data[column])
        else:
            print(f"LabelEncoder not found for column '{column}'")

# Ensure the test data matches the training feature set
X_test = test_data

# Predict using the final model
test_predictions = final_model.predict(X_test)

# Display or save predictions
output = pd.DataFrame({'Prediction': test_predictions})
print(output)

# Save predictions to a CSV file
output.to_csv('test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'") """