import numpy as np
import matplotlib.pyplot as plt

# Define the range of time steps T
T = np.arange(1, 1001)

# Number of arms K for UCB1 and Exp3
K = 5
delta = 0.1

# Regret bounds
regret_etc = T  # O(T)
regret_ucbd = T  # O(T)
regret_exp3 = np.sqrt(K * T * np.log(K))  # O(sqrt(K*T*log(K)))

# Plot all three regret functions on the same graph
plt.plot(T, regret_etc, label='ETC')
plt.plot(T, regret_ucbd, label='UCB Delta')
plt.plot(T, regret_exp3, label='EXP3')

# Add labels and title
plt.xlabel('Time Steps (T)')
plt.ylabel('Regret')
plt.title('Regret Bounds of ETC, UCB, and EXP3 Algorithms on a Adversarial Dataset')

# Add grid lines for better readability
plt.grid(True)

# Show the legend
plt.legend()

# Display the plot
plt.show()
