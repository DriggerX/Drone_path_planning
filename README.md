
# Q-Learning-Based Path Planning

---

## 🧠 Project Overview
This project demonstrates how **Q-learning**, a reinforcement learning algorithm, can be used to teach an agent (robot/drone) to find the shortest path in both **2D** and **3D** grid environments with obstacles. The implementation includes:

- A grid-based 2D environment with static obstacles.
- An extended 3D environment with multi-storey building blocks.
- A modular Python codebase to simulate, train, and visualize the agent's learning process.

---

## 🗺️ Problem Statement
Given a start point and a goal point in a grid with obstacles, the objective is to train an agent to learn the optimal policy (shortest collision-free path) using **model-free reinforcement learning (Q-learning)**.

---

## ⚙️ Q-Learning Algorithm

### 🔢 Core Update Rule:
```
Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

### Parameters:

| Parameter              | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `α` (Learning rate)    | How fast the Q-table updates                   |
| `γ` (Discount rate)    | Future reward importance                       |
| `ε` (Exploration rate) | Trade-off between exploration and exploitation |

### 🧭 Exploration Policy
An ε-greedy strategy is used with **exponential decay**:
```
ε = ε_{min} + (ε_{max} - ε_{min}) * exp(-decay * episode)
```

---

## 🧩 Environment Setup

### ✅ 2D Environment
- Grid: 10x10
- Obstacles: Predefined static blocks
- Actions: [up, down, left, right]
- State: (x, y)
- Q-table Shape: [10, 10, 4]

### ✅ 3D Environment
- Grid: 10x10x10
- Obstacles: Buildings with variable heights
- Actions: [up, down, left, right, up_z, down_z]
- State: (x, y, z)
- Q-table Shape: [10, 10, 10, 6]

---

## 🔄 Workflow & Flowchart

### **Flowchart:**
```
   +------------------+
   | Initialize Q-Table|
   +------------------+
             |
             v
   +------------------+
   | Set environment  |
   +------------------+
             |
             v
   +--------------------------+
   | For each episode:        |
   |  - Choose action (explore/exploit) |
   |  - Move to next state    |
   |  - Get reward            |
   |  - Update Q-value        |
   |  - Decay ε              |
   +--------------------------+
             |
             v
   +------------------+
   | Use learned Q-Table |
   | for path testing   |
   +------------------+
             |
             v
   +----------------------+
   | Visualize path & grid |
   +----------------------+
```

---

## 🐍 Code Structure
```
.
├── rl_path_optimized.py      # 2D Q-learning implementation
├── rl_3dpath.py              # 3D Q-learning implementation
├── images/
│   └── environment_diagram.jpg  # Visual map layout
├── plots/                    # Generated result plots
├── README.md                 # Project overview (this file)
└── wiki/                     # Optional extended documentation
```

---

## 📊 Results

### 2D Environment:
- Agent learns to move around complex obstacles to reach the goal.
- Optimal path plotted using heatmap.

### 3D Environment:
- Agent learns vertical and horizontal maneuvers.
- 3D bar plots visualize both buildings and path.

---

## 📈 Visualization Samples
- `matplotlib` used for 3D path and bar charts.
- `seaborn.heatmap` for 2D path visualization.

---

## 🚀 Future Work
- Dynamic obstacle support.
- Continuous space learning via Deep Q Networks (DQN).
- Real drone integration using ROS or PX4 flight controller.

---

## 🙌 Credits
- Developed by: [Your Name]
- Tools used: Python, NumPy, Matplotlib, Seaborn
- Special thanks to OpenAI’s support tools for documentation.

---

## 📎 Related Files
- `rl_path_optimized.py`
- `rl_3dpath.py`
- 2D grid design (image)
- Q-table output plots

---

> For any questions, open an [Issue](https://github.com/your-repo/issues) or reach out to the maintainer.
