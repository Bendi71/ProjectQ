"""
Project Title: Reinforcement Learning and Q-Learning in GridWorld  This project is a Python-based implementation of reinforcement learning, specifically Q-Learning, in a grid-based environment known as GridWorld. The GridWorld is a simple environment where an agent can move in four directions: up, down, left, and right. The world is populated with obstacles that the agent must avoid, and a goal that the agent must reach.  The agent is controlled by a Q-Learning algorithm. Q-Learning is a model-free reinforcement learning algorithm that seeks to find the best action to take given the current state. It's considered model-free because it doesn't require knowledge about the environment and how it behaves. The agent learns from its experience by updating the Q-values, which represent the expected future rewards for each action in each state.  The Q-Learning agent is initialized with a state size, action size, learning rate, discount factor, and an epsilon value for the epsilon-greedy policy. The agent uses the epsilon-greedy policy to balance exploration and exploitation. With a probability of epsilon, the agent chooses a random action (exploration), and the rest of the time it chooses the action with the highest expected future reward (exploitation).  The agent learns by interacting with the environment. For each action it takes, it receives a reward and updates the Q-value of the taken action based on the received reward and the maximum Q-value of the next state. This process is repeated for a number of episodes, with each episode consisting of a sequence of actions until the goal is reached or a maximum number of steps is taken.  The project includes a visualization of the GridWorld environment, where the agent's position, the goal, and the obstacles are displayed. After the training phase, the agent's performance is tested in the environment, and the result is visualized.  This project provides a practical example of how Q-Learning can be used to train an agent to navigate in a simple environment. It can be extended to more complex environments and other reinforcement learning algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridWorld:
    def __init__(self, grid_size, start, goal, obstacles):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.reset()

    def reset(self):
        self.agent_position = self.start
        return self.agent_position

    def step(self, action):
        # Define actions: 0 = up, 1 = down, 2 = left, 3 = right
        action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_position = (self.agent_position[0] + action_effects[action][0],
                        self.agent_position[1] + action_effects[action][1])

        # Check if the new position is valid
        if (0 <= new_position[0] < self.grid_size[0] and
                0 <= new_position[1] < self.grid_size[1] and
                new_position not in self.obstacles):
            self.agent_position = new_position

        # Check if the goal is reached
        if self.agent_position == self.goal:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False

        return self.agent_position, reward, done

    def render(self, title="Grid World"):
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0.5, self.grid_size[0], 1))
        ax.set_yticks(np.arange(0.5, self.grid_size[1], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        for obs in self.obstacles:
            rect = patches.Rectangle((obs[1], obs[0]), 1, 1, linewidth=1, edgecolor='k', facecolor='black')
            ax.add_patch(rect)

        rect = patches.Rectangle((self.goal[1], self.goal[0]), 1, 1, linewidth=1, edgecolor='k', facecolor='green')
        ax.add_patch(rect)

        rect = patches.Rectangle((self.agent_position[1], self.agent_position[0]), 1, 1, linewidth=1, edgecolor='k',
                                 facecolor='blue')
        ax.add_patch(rect)

        plt.gca().invert_yaxis()
        plt.title(title)
        plt.show()

labyrinth_obstacles = [
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (3, 0), (3, 1), (3, 3), (3, 4), (3, 5), (3, 6), (3, 8), (3, 9),
    (5, 1), (5, 2), (5, 4), (5, 6), (5, 8), (5, 9),
    (7, 0), (7, 1), (7, 3), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9),
    (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9)
]


# Initialize the environment
env = GridWorld(grid_size=(10, 10), start=(0, 0), goal=(9, 0), obstacles=labyrinth_obstacles)
env.render()


class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros(state_size + (action_size,))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# Training parameters
num_episodes = 1000
max_steps_per_episode = 40

# Initialize the agent
agent = QLearningAgent(state_size=env.grid_size, action_size=4)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state = tuple(state)
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        next_state = tuple(next_state)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

print("Training finished!")

# Test the trained agent
state = env.reset()
state = tuple(state)

for step in range(max_steps_per_episode):
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    next_state = tuple(next_state)
    state = next_state
    env.render()

    if done:
        print(f"Goal reached in {step + 1} steps!")
        break
