import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
grid_size = 10
start = (0, 0)
finish = (9,9)
actions = ['up', 'down', 'left', 'right']
num_episodes = 2000
max_steps_per_episode = 200
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

grid = np.zeros((grid_size, grid_size))

# Obstacles: Add 2x2, 4x4, and 2 additional obstacles
obstacles = []
obstacles.extend([(2, 2), (2, 3), (3, 2), (3, 3)]) 
obstacles.extend([(5 + x, 5 + y) for x in range(4) for y in range(4)]) 
obstacles.extend([(1, 5), (2, 6), (1,6 ),(2,5)])  
obstacles.extend([(7, 2), (8, 2),(8,3),(7,3)])  
for obs in obstacles:
    grid[obs] = -1

q_table = np.zeros((grid_size, grid_size, len(actions)))

# Helper functions
def is_valid_position(position):
    return (
        0 <= position[0] < grid_size
        and 0 <= position[1] < grid_size
        and position not in obstacles
    )

def get_next_position(position, action):
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

        if random.uniform(0, 1) < exploration_rate:
            action_index = random.randint(0, len(actions) - 1)  # Explore
        else:
            action_index = np.argmax(q_table[state[0], state[1], :])  # Exploit
        
        action = actions[action_index]
        next_state = get_next_position(state, action)
        
        # Reward system
        if next_state == finish:
            reward = 100  # High reward for reaching the goal
        elif next_state in obstacles:
            reward = -100  # Penalty for hitting obstacles
        else:
            reward = -1 #for each step optimization = we will want to increase the rewards for the best solution
        
        q_table[state[0], state[1], action_index] = q_table[state[0], state[1], action_index] + \
            learning_rate * (reward + discount_rate * np.max(q_table[next_state[0], next_state[1], :]) - \
                             q_table[state[0], state[1], action_index])
        
        state = next_state
        
        if state == finish:
            break

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode) #updating with decay exploration rate

# Testing
state = start
path = [state]
while state != finish:
    action_index = np.argmax(q_table[state[0], state[1], :])
    action = actions[action_index]
    state = get_next_position(state, action)
    path.append(state)

# Mark path in the grid for visualization
for pos in path:
    if pos != start and pos != finish:
        grid[pos] = 1  # Mark the path

plt.figure(figsize=(10, 8))
cmap = sns.color_palette("coolwarm", as_cmap=True)

visual_grid = np.zeros_like(grid)
visual_grid[start] = 0.5 
visual_grid[finish] = 0.8  
for obs in obstacles:
    visual_grid[obs] = -0.5  
for pos in path:
    if pos != start and pos != finish:
        visual_grid[pos] = 1  # Path

sns.heatmap(visual_grid, annot=False, cmap=cmap, cbar=False, linewidths=0.5, linecolor='black')
plt.title("Shortest Path with Q-learning", fontsize=16)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

print("Shortest Path Taken by the Agent:", path)
