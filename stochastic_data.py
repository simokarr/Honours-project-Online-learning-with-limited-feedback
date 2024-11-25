import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Arm definition
arms = [
    {'mean': 0.9, 'std': 0.2},  # Best arm
    {'mean': 0.7, 'std': 0.3},  # Second best arm
    {'mean': 0.4, 'std': 0.4},  # Medium arm
    {'mean': 0.2, 'std': 0.2},  # Poor arm
    {'mean': 0.1, 'std': 0.1}   # Worst arm
]

def generate_rewards(arms, num_rounds):
    """Generate rewards for each arm across num_rounds"""
    rewards = np.zeros((len(arms), num_rounds))
    for i, arm in enumerate(arms):
        rewards[i] = np.random.normal(
            loc=arm['mean'], 
            scale=arm['std'], 
            size=num_rounds
        )
    return rewards

def true_best_arm(arms):
    """Find the index of the arm with highest mean"""
    return max(range(len(arms)), key=lambda i: arms[i]['mean'])

def explore_then_commit(arms, num_rounds, explore_rounds):
    """
    Explore Then Commit (ETC) Algorithm
    
    Args:
    - arms: List of arm dictionaries
    - num_rounds: Total number of rounds
    - explore_rounds: Number of rounds for exploration
    
    Returns:
    - Cumulative rewards
    - Cumulative regret
    """
    num_arms = len(arms)
    best_arm_index = true_best_arm(arms)
    best_arm_mean = arms[best_arm_index]['mean']
    
    # Rewards array
    rewards_array = generate_rewards(arms, num_rounds)
    
    # Tracking variables
    cumulative_reward = 0
    cumulative_regret = 0
    regret_history = np.zeros(num_rounds)
    
    # Exploration phase: sample each arm equally
    for t in range(explore_rounds):
        arm_index = t % num_arms
        reward = rewards_array[arm_index, t]
        cumulative_reward += reward
        
        # Calculate regret
        round_regret = best_arm_mean - reward
        cumulative_regret += round_regret
        regret_history[t] = cumulative_regret
    
    # Exploitation phase: play best arm found during exploration
    best_arm_during_explore = np.argmax([
        np.mean(rewards_array[i, :explore_rounds]) 
        for i in range(num_arms)
    ])
    
    for t in range(explore_rounds, num_rounds):
        reward = rewards_array[best_arm_during_explore, t]
        cumulative_reward += reward
        
        # Calculate regret
        round_regret = best_arm_mean - reward
        cumulative_regret += round_regret
        regret_history[t] = cumulative_regret
    
    return cumulative_reward, regret_history

def ucb1(arms, num_rounds):
    """
    Upper Confidence Bound (UCB1) Algorithm
    
    Args:
    - arms: List of arm dictionaries
    - num_rounds: Total number of rounds
    
    Returns:
    - Cumulative rewards
    - Cumulative regret
    """
    num_arms = len(arms)
    best_arm_index = true_best_arm(arms)
    best_arm_mean = arms[best_arm_index]['mean']
    
    # Rewards array
    rewards_array = generate_rewards(arms, num_rounds)
    
    # Tracking variables
    arm_counts = np.zeros(num_arms)
    arm_means = np.zeros(num_arms)
    cumulative_reward = 0
    cumulative_regret = 0
    regret_history = np.zeros(num_rounds)
    
    # Initial exploration of each arm once
    for i in range(num_arms):
        reward = rewards_array[i, i]
        arm_counts[i] = 1
        arm_means[i] = reward
        cumulative_reward += reward
        
        # Calculate regret
        round_regret = best_arm_mean - reward
        cumulative_regret += round_regret
        regret_history[i] = cumulative_regret
    
    # UCB1 selection
    for t in range(num_arms, num_rounds):
        # UCB1 selection rule
        ucb_values = arm_means + np.sqrt(2 * np.log(t) / (arm_counts + 1e-10))
        chosen_arm = np.argmax(ucb_values)
        
        # Get reward
        reward = rewards_array[chosen_arm, t]
        
        # Update arm statistics
        arm_counts[chosen_arm] += 1
        arm_means[chosen_arm] += (reward - arm_means[chosen_arm]) / arm_counts[chosen_arm]
        
        cumulative_reward += reward
        
        # Calculate regret
        round_regret = best_arm_mean - reward
        cumulative_regret += round_regret
        regret_history[t] = cumulative_regret
    
    return cumulative_reward, regret_history

def exp3(arms, num_rounds, gamma=0.1):
    """
    Exponential Weights Algorithm for Exploration and Exploitation (EXP3)
    
    Args:
    - arms: List of arm dictionaries
    - num_rounds: Total number of rounds
    - gamma: Exploration parameter (controls exploration vs exploitation)
    
    Returns:
    - Cumulative rewards
    - Cumulative regret
    """
    num_arms = len(arms)
    best_arm_index = true_best_arm(arms)
    best_arm_mean = arms[best_arm_index]['mean']
    
    # Rewards array
    rewards_array = generate_rewards(arms, num_rounds)
    
    # Tracking variables
    weights = np.ones(num_arms)
    cumulative_reward = 0
    cumulative_regret = 0
    regret_history = np.zeros(num_rounds)
    
    for t in range(num_rounds):
        # Compute probabilities
        total_weight = np.sum(weights)
        probabilities = (1 - gamma) * (weights / total_weight) + (gamma / num_arms)
        
        # Choose arm
        chosen_arm = np.random.choice(num_arms, p=probabilities)
        
        # Get reward
        reward = rewards_array[chosen_arm, t]
        
        # Estimated reward (for stochastic case, we use the actual reward)
        estimated_reward = reward / probabilities[chosen_arm]
        
        # Update weights
        weights[chosen_arm] *= np.exp(estimated_reward * gamma / num_arms)
        
        cumulative_reward += reward
        
        # Calculate regret
        round_regret = best_arm_mean - reward
        cumulative_regret += round_regret
        regret_history[t] = cumulative_regret
    
    return cumulative_reward, regret_history

# Simulation parameters
num_rounds = 1000
explore_rounds = 100

# Run algorithms
etc_reward, etc_regret = explore_then_commit(arms, num_rounds, explore_rounds)
ucb_reward, ucb_regret = ucb1(arms, num_rounds)
exp3_reward, exp3_regret = exp3(arms, num_rounds)

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(etc_regret, label='ETC Cumulative Regret')
plt.plot(ucb_regret, label='UCB1 Cumulative Regret')
plt.plot(exp3_regret, label='EXP3 Cumulative Regret')
plt.title('Cumulative Regret Comparison')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print final regret
print(f"ETC Final Cumulative Regret: {etc_regret[-1]}")
print(f"UCB1 Final Cumulative Regret: {ucb_regret[-1]}")
print(f"EXP3 Final Cumulative Regret: {exp3_regret[-1]}")
