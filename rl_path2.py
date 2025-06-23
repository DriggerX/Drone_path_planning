import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
grid_size = 10
start = (0, 0)
finish = (9, 9)
actions = ['up', 'down', 'left', 'right']
num_episodes = 2000
max_steps_per_episode = 200
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Define the grid and obstacles
grid = np.zeros((grid_size, grid_size))

# Obstacles: Add 2x2, 4x4, and two additional custom obstacles
obstacles = []
# 2x2 square obstacle
obstacles.extend([(2, 2), (2, 3), (3, 2), (3, 3)])
# 4x4 square obstacle
obstacles.extend([(5 + x, 5 + y) for x in range(4) for y in range(4)])
# Additional obstacles (diagonal and random placement)
obstacles.extend([(1, 5), (2, 6), (1,6 ),(2,5)])  # Diagonal obstacle
obstacles.extend([(7, 2), (8, 2),(8,3),(7,3)])  # Randomly placed obstacle

# Mark obstacles in the grid
for obs in obstacles:
    grid[obs] = -1

# Initialize Q-table
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Helper functions
def is_valid_position(position):
    """Check if a position is valid (inside the grid and not an obstacle)."""
    return (
        0 <= position[0] < grid_size
        and 0 <= position[1] < grid_size
        and position not in obstacles
    )

def get_next_position(position, action):
    """Get the next position based on the current position and action."""
    if action == 'up':
        next_position = (position[0] - 1, position[1])
    elif action == 'down':
        next_position = (position[0] + 1, position[1])
    elif action == 'left':
        next_position = (position[0], position[1] - 1)
    elif action == 'right':
        next_position = (position[0], position[1] + 1)
    return next_position if is_valid_position(next_position) else position

# Training
for episode in range(num_episodes):
    state = start
    for step in range(max_steps_per_episode):
        # Exploration vs. exploitation
        if random.uniform(0, 1) < exploration_rate:
            action_index = random.randint(0, len(actions) - 1)  # Explore
        else:
            action_index = np.argmax(q_table[state[0], state[1], :])  # Exploit
        
        action = actions[action_index]
        next_state = get_next_position(state, action)
        
        # Reward system
        if next_state == finish:
            reward = 100
        elif next_state in obstacles:
            reward = -100
        else:
            reward = -1  # Penalize each step
        
        # Update Q-value
        q_table[state[0], state[1], action_index] = q_table[state[0], state[1], action_index] + \
            learning_rate * (reward + discount_rate * np.max(q_table[next_state[0], next_state[1], :]) - \
                             q_table[state[0], state[1], action_index])
        
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
    action_index = np.argmax(q_table[state[0], state[1], :])
    action = actions[action_index]
    state = get_next_position(state, action)
    path.append(state)

print("Path taken by the agent:", path)

# Mark the path on the grid
for pos in path:
    if pos != start and pos != finish:
        grid[pos] = 1  # Mark the path

# Visualization with matplotlib and seaborn
visual_grid = np.copy(grid)
visual_grid[start] = 2  # Start
visual_grid[finish] = 3  # Finish

# Define a custom color map
cmap = sns.color_palette(['white', 'green', 'blue', 'red', 'black'])
plt.figure(figsize=(8, 8))
sns.heatmap(
    visual_grid,
    annot=False,
    fmt="",
    cmap=cmap,
    cbar=False,
    linewidths=0.5,
    linecolor="gray",
    xticklabels=False,
    yticklabels=False,
)
plt.title("Path Planning with Obstacles", fontsize=16)
plt.show()
