"""
Tic-Tac-Toe with Deep Reinforcement Learning

The project is a Deep Reinforcement Learning (DRL) implementation of the game Tic-Tac-Toe. Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize some notion of cumulative reward. The agent learns from the consequences of its actions, rather than from being explicitly taught and it selects its actions based on its past experiences (exploitation) and also by new choices (exploration), which is the dilemma in reinforcement learning.  Deep Reinforcement Learning (DRL) is the combination of reinforcement learning and deep learning. It uses a neural network to approximate the reward function and update the network weights using gradient descent. The neural network takes in the state of the game as input and outputs the corresponding Q-values for each action in the state.  A Deep Q-Network (DQN) is a type of DRL where the Q-function is approximated using a deep neural network. The Q-function takes a state-action pair and returns the expected future reward of that action taken in that state. DQN uses a technique called experience replay where past state transitions are stored and during training, a mini-batch of these transitions is used to update the network weights. This helps to break the correlation between consecutive states and stabilize the training process.  The project consists of three main parts:

The TicTacToe class: This class represents the Tic-Tac-Toe game environment. It has methods to print the game board, check the available actions, check if a player has won, check if the game is a draw, and play an action.

The QNetwork class: This class represents the neural network used to approximate the Q-function. It has three fully connected layers and uses the ReLU activation function.

The DQNAgent class: This class represents the DQN agent. It has methods to choose an action (either randomly or based on the Q-values predicted by the network), convert the game state to input for the network, remember a state transition, replay the past transitions and update the network weights, and update the target network.

The train_dqn function is used to train the DQN agents. It alternates between the 'X' and 'O' agents, gets the chosen action, plays the action, checks the game status, remembers the state transition, and updates the network weights. The target network is updated periodically.  The trained network weights are saved to a file at the end of training.

Unfortunetely, the learning did not work as I expected. Originally I thought that as the agents play more game,
they will play more rational with fewer mistakes, thus they will always end up in a draw. However, draws rarely
dominated the outcome. Most of the time, it finished with an equilibirum where X wins the most and draws are the
least frequent in outcomes.I think the reason for this is that the network's hyperparameters are not tuned properly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

# Define Tic-Tac-Toe environment
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # Initial empty board

    def print_board(self):
        print('-------------')
        for i in range(3):
            print('| ' + ' | '.join(self.board[i*3:i*3+3]) + ' |')
            print('-------------')

    def available_actions(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def is_winner(self, player):
        # Check rows, columns, and diagonals for a win
        win_states = [[0, 1, 2], [3, 4, 5], [6, 7, 8],
                      [0, 3, 6], [1, 4, 7], [2, 5, 8],
                      [0, 4, 8], [2, 4, 6]]
        for state in win_states:
            if all(self.board[i] == player for i in state):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def play(self, action, player):
        next_state = TicTacToe()
        next_state.board = self.board.copy()
        next_state.board[action] = player
        return next_state

# Define Q-Network
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define DQN Agent
class DQNAgent:
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)  # Use deque for memory management
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

    def choose_action(self, state):
        available_actions = state.available_actions()
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(self.state_to_input(state)))
                q_values = q_values.squeeze()
                available_q_values = q_values[available_actions]
                return available_actions[torch.argmax(available_q_values).item()]

    def state_to_input(self, state):
        player = 'X' if state.board.count('X') == state.board.count('O') else 'O'
        one_hot_board = [1 if cell == player else -1 if cell != ' ' else 0 for cell in state.board]
        return np.array(one_hot_board + [1 if player == 'X' else -1]).reshape(1, -1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.vstack([self.state_to_input(s) for s in states]))
        next_states = torch.FloatTensor(np.vstack([self.state_to_input(s) for s in next_states]))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Train DQN Agent
def train_dqn(agent_x, agent_o, episodes, target_update_frequency=7):
    outcomes = {'X': 0, 'O': 0, 'Draw': 0}
    steps = {str(i): 0 for i in range(9)}

    for episode in range(episodes):
        env = TicTacToe()  # Reset the environment for each episode
        state = env
        done = False
        if episode % 2 == 0:
            current_player = 'X'
        else:
            current_player = 'O'

        while not done:
            if current_player == 'O':
                action = agent_o.choose_action(state)
            else:
                action = agent_x.choose_action(state)
                steps[str(action)] += 1

            next_state = state.play(action, current_player)

            if next_state.is_winner(current_player):
                reward_currentplayer = 20
                reward_otherplayer = -20
                outcomes['X'] += 1 if current_player == 'X' else 0
                outcomes["O"] += 1 if current_player == 'O' else 0

                if current_player == 'X':
                    agent_x.remember(state, action, reward_currentplayer, next_state, True)
                    agent_o.remember(state, action, reward_otherplayer, next_state, True)
                else:
                    agent_o.remember(state, action, reward_currentplayer, next_state, True)
                    agent_x.remember(state, action, reward_otherplayer, next_state, True)
                done = True
            elif next_state.is_draw():
                reward = 3
                outcomes["Draw"] += 1
                agent_x.remember(state, action, reward, next_state, True)
                agent_o.remember(state, action, reward, next_state, True)
                done = True
            else:
                reward = 0
                agent_x.remember(state, action, reward, next_state, False)
                agent_o.remember(state, action, reward, next_state, False)

            state = next_state
            current_player = 'O' if current_player == 'X' else 'X'

            if len(agent_x.memory) >= agent_x.batch_size:
                agent_x.replay()
            if len(agent_o.memory) >= agent_o.batch_size:
                agent_o.replay()

        if episode % target_update_frequency == 0:
            agent_x.update_target_network()
            agent_o.update_target_network()

        if episode % 100 == 0:
            print(f'Episode {episode}, epsilon_x: {agent_x.epsilon:.4f}, epsilon_o: {agent_o.epsilon:.4f}')
            for k, v in outcomes.items():
                print(f'{k}: {v / sum(outcomes.values()) * 100:.2f}%')
            #for k, v in steps.items():
            #    print(f'{k}: {v / sum(steps.values()) * 100:.2f}%')
            #steps = {str(i): 0 for i in range(9)}

if __name__ == '__main__':
    agent_x = DQNAgent()
    agent_o = DQNAgent()
    train_dqn(agent_x, agent_o, episodes=10001)
    print('Training complete')
    torch.save(agent_x.q_network.state_dict(), 'model_x.pth')
    torch.save(agent_o.q_network.state_dict(), 'model_o.pth')
    print('Models saved')



