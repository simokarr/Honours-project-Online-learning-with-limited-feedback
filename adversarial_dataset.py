import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
n_arms = 5
T = 10000
variance = 0.1
runs = 10

# Determine optimal delta
initial_means = [0.1, 0.2, 0.3, 0.25, 0.15]
initial_best_mean = max(initial_means)
minimal_difference = min(abs(initial_best_mean - mean) for mean in initial_means if mean != initial_best_mean)
delta = minimal_difference / 2


class EXP3:
    def __init__(self, n_arms, gamma=0.05):
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

class ETC:
    def __init__(self, n_arms, m=200):
        self.n_arms = n_arms
        self.m = m
        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.total_rewards = np.zeros(n_arms)
        self.committed_arm = None  # Arm to commit to after exploration
    
    def select_arm(self):
        if np.any(self.pull_counts < self.m):
            candidate_arms = np.where(self.pull_counts < self.m)[0]
            selected_arm = np.random.choice(candidate_arms)
        else:
            if self.committed_arm is None:
                empirical_means = self.total_rewards / self.pull_counts
                self.committed_arm = np.argmax(empirical_means)
            selected_arm = self.committed_arm
        return selected_arm
    
    def update(self, chosen_arm, reward):
        self.pull_counts[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward

class UCB_delta:
    def __init__(self, n_arms, delta):
        self.n_arms = n_arms
        self.delta = delta
        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.total_rewards = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0

    def select_arm(self):
        if self.total_pulls < self.n_arms:
            selected_arm = np.argmin(self.pull_counts)
        else:
            mean_rewards = self.total_rewards / self.pull_counts
            # Calculate the confidence bound for UCB-delta
            confidence_bound = np.sqrt((1 / (2 * self.pull_counts)) * np.log(1 / self.delta))
            ucb = mean_rewards + confidence_bound
            selected_arm = np.argmax(ucb)
        return selected_arm

    def update(self, chosen_arm, reward):
        self.pull_counts[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward
        self.total_pulls += 1

def run_simulation(alg, n_arms, runs=10, T=10000, means_change_time=int(T/2)):
    cumulative_regret = np.zeros(T)
    for _ in range(runs):
        regret = 0.0
        alg.reset()
        means = [0.1, 0.2, 0.3, 0.25, 0.15]  # Initial means
        for t in range(T):
            if isinstance(alg, EXP3):
                arm, probabilities = alg.select_arm()
            else:
                arm = alg.select_arm()
            # Change the means after a certain time step
            if t >= means_change_time:
                new_means = [0.3, 0.15, 0.2, 0.25, 0.1]  # Adversarial change
                mean = new_means[arm]
            else:
                mean = means[arm]
            reward = np.random.normal(mean, np.sqrt(variance))
            if isinstance(alg, EXP3):
                alg.update(arm, reward, probabilities)
            else:
                alg.update(arm, reward)
            best_mean = max(means if t < means_change_time else new_means)
            regret += best_mean - mean
            cumulative_regret[t] += regret
    cumulative_regret /= runs
    return cumulative_regret

# Ensure algorithms have a reset method
class EXP3WithReset(EXP3):
    def reset(self):
        self.weights = np.ones(self.n_arms)

class ETCWithReset(ETC):
    def reset(self):
        self.pull_counts = np.zeros(self.n_arms, dtype=int)
        self.total_rewards = np.zeros(self.n_arms)
        self.committed_arm = None  # Arm to commit to after exploration

class UCB_delta_with_Reset(UCB_delta):
    def reset(self):
        self.pull_counts = np.zeros(self.n_arms, dtype=int)
        self.total_rewards = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0


# Run simulations
exp3 = EXP3WithReset(n_arms)
cumulative_regret_exp3 = run_simulation(exp3, n_arms, runs, T)

etc = ETCWithReset(n_arms)
cumulative_regret_etc = run_simulation(etc, n_arms, runs, T)

ucb_delta = UCB_delta_with_Reset(n_arms, delta)
cumulative_regret_ucb_delta = run_simulation(ucb_delta, n_arms, runs, T)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_regret_exp3, label='EXP3', color='blue')
plt.plot(cumulative_regret_etc, label='ETC', color='green')
plt.plot(cumulative_regret_ucb_delta, label='UCB-delta', color='red')
plt.title('Average Cumulative Regret Over Time with Adversarial Reward Changes')
plt.xlabel('Time Steps')
plt.ylabel('Average Cumulative Regret')
plt.legend()
plt.grid(True)
plt.show()
