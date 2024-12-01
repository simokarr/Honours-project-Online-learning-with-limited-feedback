import numpy as np

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

class ETC:
    def __init__(self, n_arms, m=100):
        self.n_arms = n_arms
        self.m = m
        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.total_rewards = np.zeros(n_arms)
        self.committed_arm = None  # Arm to commit to after exploration
    
    def select_arm(self):
        # Exploration phase: pull each arm m times
        if np.any(self.pull_counts < self.m):
            # Find arms that have been pulled fewer than m times
            candidate_arms = np.where(self.pull_counts < self.m)[0]
            # Uniformly select one of the candidate arms
            selected_arm = np.random.choice(candidate_arms)
        else:
            # Commitment phase: choose the arm with the highest empirical mean
            if self.committed_arm is None:
                # Compute empirical means
                empirical_means = self.total_rewards / self.pull_counts
                # Select the arm with the highest empirical mean
                # In case of ties, select the one with the smallest index
                self.committed_arm = np.argmax(empirical_means)
            selected_arm = self.committed_arm
        return selected_arm
    
    def update(self, chosen_arm, reward):
        self.pull_counts[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.total_rewards = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0
    
    def select_arm(self):
        if self.total_pulls < self.n_arms:
            # Initialization phase: pull each arm at least once
            selected_arm = np.argmin(self.pull_counts)
        else:
            # Compute mean rewards
            mean_rewards = self.total_rewards / self.pull_counts
            # Compute exploration term
            ln_t = np.log(self.total_pulls)
            exploration = np.sqrt((2 * ln_t) / self.pull_counts)
            # Compute UCB for each arm
            ucb = mean_rewards + exploration
            # Select the arm with the highest UCB
            selected_arm = np.argmax(ucb)
        return selected_arm
    
    def update(self, chosen_arm, reward):
        self.pull_counts[chosen_arm] += 1
        self.total_rewards[chosen_arm] += reward
        self.total_pulls += 1
