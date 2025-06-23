import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
grid_size = 10
start = (0, 0, 0)
finish = (9, 9, 9)
actions = ['up', 'down', 'left', 'right', 'up_z', 'down_z']
num_episodes = 3000
max_steps_per_episode = 300
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Define the 3D grid
grid = np.zeros((grid_size, grid_size, grid_size))

# Define buildings with varying heights
buildings = [
    {"base": [(3, 3), (6, 6)], "height": 4},  # Building 1
    {"base": [(1, 1), (2, 2)], "height": 6},  # Building 2
    {"base": [(7, 7), (9, 9)], "height": 5},  # Building 3
]

# Mark buildings in the grid
for building in buildings:
    x_min, y_min = building["base"][0]
    x_max, y_max = building["base"][1]
    height = building["height"]
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            for z in range(height):
                grid[x, y, z] = -1  # Mark as obstacle

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, grid_size, len(actions)))

# Helper functions
def is_valid_position(position):
    """Check if a position is valid (inside the grid and not an obstacle)."""
    return (
        0 <= position[0] < grid_size
        and 0 <= position[1] < grid_size
        and 0 <= position[2] < grid_size
        and grid[position[0], position[1], position[2]] != -1
    )

def get_next_position(position, action):
    """Get the next position based on the current position and action."""
    if action == 'up':
        next_position = (position[0] - 1, position[1], position[2])
    elif action == 'down':
        next_position = (position[0] + 1, position[1], position[2])
    elif action == 'left':
        next_position = (position[0], position[1] - 1, position[2])
    elif action == 'right':
        next_position = (position[0], position[1] + 1, position[2])
    elif action == 'up_z':
        next_position = (position[0], position[1], position[2] + 1)
    elif action == 'down_z':
        next_position = (position[0], position[1], position[2] - 1)
    return next_position if is_valid_position(next_position) else position

# Training
for episode in range(num_episodes):
    state = start
    for step in range(max_steps_per_episode):
        # Exploration vs. exploitation
        if random.uniform(0, 1) < exploration_rate:
            action_index = random.randint(0, len(actions) - 1)  # Explore
        else:
            action_index = np.argmax(q_table[state[0], state[1], state[2], :])  # Exploit
        
        action = actions[action_index]
        next_state = get_next_position(state, action)
        
        # Reward system
        if next_state == finish:
            reward = 100  # High reward for reaching the goal
        elif grid[next_state[0], next_state[1], next_state[2]] == -1:
            reward = -100  # Penalty for hitting buildings
        else:
            reward = -1  # Penalize each step to encourage shorter paths
        
        # Update Q-value
        q_table[state[0], state[1], state[2], action_index] = q_table[state[0], state[1], state[2], action_index] + \
            learning_rate * (reward + discount_rate * np.max(q_table[next_state[0], next_state[1], next_state[2], :]) - \
                             q_table[state[0], state[1], state[2], action_index])
        
        state = next_state
        
        if state == finish:
            break

    # Decay exploration rate
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

# Test the learned policy
state = start
path = [state]
while state != finish:
    action_index = np.argmax(q_table[state[0], state[1], state[2], :])
    action = actions[action_index]
    state = get_next_position(state, action)
    path.append(state)

# Visualization using matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot buildings
for building in buildings:
    x_min, y_min = building["base"][0]
    x_max, y_max = building["base"][1]
    height = building["height"]
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            ax.bar3d(x, y, 0, 1, 1, height, color='red', alpha=0.5, label='Building' if (x, y) == (x_min, y_min) else "")

# Plot the path
path_x, path_y, path_z = zip(*path)
ax.plot(path_x, path_y, path_z, c='blue', label='Path')

# Plot start and finish
ax.scatter(*start, c='green', marker='o', s=100, label='Start')
ax.scatter(*finish, c='purple', marker='x', s=100, label='Finish')

# Labels and legend
ax.set_title("3D Pathfinding with Buildings")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()

# Print the optimized path
print("Shortest Path Taken by the Agent:", path)
