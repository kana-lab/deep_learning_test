import random
from collections import deque
# from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt, animation

from util import SumTree


class ReplayBuffer:
    def __init__(self, buf_size=5000, batch_size=32):
        self.buf = deque(maxlen=buf_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buf)

    def add(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buf, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in batch]))
        action = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long)
        reward = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32)
        next_state = torch.tensor(np.stack([x[3] for x in batch]))
        done = torch.tensor(np.array([x[4] for x in batch]).astype(np.int32))
        return state, action, reward, next_state, done


# ref: https://jsapachehtml.hatenablog.com/entry/2018/09/24/225051
class PERBuffer:
    epsilon = 1e-4
    alpha = 0.6

    def __init__(self, buf_size=5000, batch_size=32):
        self.size = 0
        self.tree = SumTree(buf_size)
        self.batch_size = batch_size

    # proportional prioritization
    def _get_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        self.size += 1
        priority = self.tree.max_p
        if priority <= 0:
            priority = 1.

        self.tree.add(priority, transition)

    def sample(self):
        indices, batch = [], []
        for rand in np.random.uniform(0, self.tree.total(), self.batch_size):
            idx, _, transition = self.tree.get(rand)
            indices.append(idx)
            batch.append(transition)

        state = torch.tensor(np.stack([x[0] for x in batch]))
        action = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long)
        reward = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32)
        next_state = torch.tensor(np.stack([x[3] for x in batch]))
        done = torch.tensor(np.array([x[4] for x in batch]).astype(np.int32))
        return indices, state, action, reward, next_state, done

    # 引数はテンソルを想定
    def update(self, idx, td_error):
        priority = self._get_priority(td_error)
        for (i, p) in zip(idx, priority):
            self.tree.update(i, p)

    def __len__(self):
        return self.size


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DuelQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.v1 = nn.Linear(256, 64)
        self.v2 = nn.Linear(64, 1)
        self.adv1 = nn.Linear(256, 128)
        self.adv2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = F.relu(self.v1(x))
        v = F.relu(self.v2(v))
        adv = F.relu(self.adv1(x))
        adv = F.relu(self.adv2(adv))

        with torch.no_grad():
            mean = torch.mean(adv, 1, keepdim=True)

        q = v + adv - mean
        return q


class Agent:
    def __init__(self, epsilon=0.1, gamma=0.98, batch_size=32):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.net = QNet()
        self.target_net = QNet()
        self.optimizer = optim.Adam(self.net.parameters())
        self.replay_buf = ReplayBuffer(batch_size=batch_size)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(2)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.net(state)
            return qs.argmax().item()

    def sync_net(self):
        # self.target_net = deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())

    def update(self, state, action, reward, next_state, done):
        self.replay_buf.add(state, action, reward, next_state, done)
        if len(self.replay_buf) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buf.sample()
        qs = self.net(state)
        q = qs[np.arange(self.batch_size), action]

        # with torch.no_grad():
        next_qs = self.target_net(next_state)
        next_q = next_qs.max(axis=1)[0]
        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQNAgent(Agent):
    def update(self, state, action, reward, next_state, done):
        self.replay_buf.add(state, action, reward, next_state, done)
        if len(self.replay_buf) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buf.sample()
        qs = self.net(state)
        q = qs[np.arange(self.batch_size), action]

        # with torch.no_grad():
        next_qs = self.net(next_state)
        max_as = next_qs.argmax(axis=1)
        next_qs_prime = self.target_net(next_state)
        next_q_prime = next_qs_prime[np.arange(self.batch_size), max_as]
        next_q_prime.detach()
        target = reward + (1 - done) * self.gamma * next_q_prime

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 重点サンプリングはやらない！
class PERDDQN(Agent):
    def __init__(self, epsilon=0.1, gamma=0.98, batch_size=32):
        super().__init__(epsilon, gamma, batch_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.net = QNet()
        self.target_net = QNet()
        self.optimizer = optim.Adam(self.net.parameters())
        self.replay_buf = PERBuffer(batch_size=batch_size)

    def update(self, state, action, reward, next_state, done):
        self.replay_buf.add(state, action, reward, next_state, done)
        if len(self.replay_buf) < self.batch_size:
            return

        idx, state, action, reward, next_state, done = self.replay_buf.sample()
        qs = self.net(state)
        q = qs[np.arange(self.batch_size), action]

        # with torch.no_grad():
        next_qs = self.net(next_state)
        max_as = next_qs.argmax(axis=1)
        next_qs_prime = self.target_net(next_state)
        next_q_prime = next_qs_prime[np.arange(self.batch_size), max_as]
        next_q_prime.detach()
        target = reward + (1 - done) * self.gamma * next_q_prime

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if idx is not None:
            delta = (target - q).detach().numpy()
            self.replay_buf.update(idx, np.abs(delta))


class PERDuelingDDQN(PERDDQN):
    def __init__(self, epsilon=0.1, gamma=0.98, batch_size=32):
        super().__init__(epsilon, gamma, batch_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.net = DuelQNet()
        self.target_net = DuelQNet()
        self.optimizer = optim.Adam(self.net.parameters())
        self.replay_buf = PERBuffer(batch_size=batch_size)


def train(env, agent: Agent):
    episodes = 300
    sync_interval = 20

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _info, _info2 = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_net()

        if episode % 10 == 0:
            print(f'episode {episode}: {total_reward}')


def main():
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    agent = PERDuelingDDQN()
    # agent = PERDDQN()
    train(env, agent)

    fig = plt.figure()
    fig.suptitle('Cart Pole', fontsize=20)
    frames = []

    agent.epsilon = 0.
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _info, _info2 = env.step(action)
        total_reward += reward
        rgb_data = env.render()

        state_text = f'cart position={state[0]:5.2f}, '
        state_text += f'cart velocity={state[1]:6.3f}\n'
        state_text += f'pole angle   ={state[2]:5.2f}, '
        state_text += f'pole velocity={state[3]:6.3f}'

        # カートポールを描画
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title(state_text, loc='left')
        img = plt.imshow(rgb_data)

        frames.append([img])
        state = next_state

    print(f'total reward: {total_reward}')
    ani = animation.ArtistAnimation(fig, frames, interval=70)
    ani.save('cart_pole_per_dueling_ddqn.gif', codec='gif')


if __name__ == '__main__':
    main()
