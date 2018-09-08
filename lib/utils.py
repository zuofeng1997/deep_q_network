import numpy as np
import collections
import torch

HYPERPARAMS = {
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'replay_size':      100000,
        'replay_start_size':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'fps':              60,
        'state_eval_num':       100,
    }
}


class PrioReplayBuffer:
    def __init__(self, buf_size, prob_alpha):
        self.capacity = buf_size
        self.prob_alpha = prob_alpha
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, exp):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        prios = prios ** self.prob_alpha
        prios /= prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=prios)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        total = len(self.buffer)
        weights = (total * prios[indices]) ** (-beta)
        weights /= weights.max()
        return (np.array(states), np.array(actions), np.array(rewards), np.array(dones, dtype=np.uint8),\
               np.array(next_states)), indices, np.array(weights, dtype=np.float32)

    def update_priorites(self, batch_indices, batch_priorites):
        for idx, prio in zip(batch_indices, batch_priorites):
            self.priorities[idx] = prio


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards), np.array(dones, dtype=np.uint8),\
               np.array(next_states)


def choose_action(env, state, net, epsilon=0.0, use_cuda=True):
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        state_t = torch.tensor(state).to(device).unsqueeze(0)
        q_value = net(state_t)
        _, action = torch.max(q_value, dim=1)
        action = action.item()
    return action


def learn(loss_fn, optimizer, batch, params, net, target_net, double=False, use_cuda=True, prio=False, batch_w=None, two_steps=False):
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if two_steps:
        gamma = params["gamma"]
    else:
        gamma = params["gamma"] * params["gamma"]
    states, actions, rewards, dones, next_states = batch
    states = torch.tensor(states, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    dones = torch.tensor(dones, dtype=torch.uint8, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float, device=device)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states).max(1)[1]
        next_state_values = target_net(next_states).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = target_net(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()
    optimizer.zero_grad()
    if prio:
        batch_w = torch.tensor(batch_w).to(device)
        loss_ind = batch_w*(state_action_values-rewards + gamma * next_state_values)**2
        loss = loss_ind.mean()
        loss.backward()
        optimizer.step()
        return loss_ind
    else:
        loss = loss_fn(state_action_values, rewards + gamma * next_state_values)
        loss.backward()
        optimizer.step()






def logger(writer, frame_idx, reward):
    writer.add_scalar("reward", reward, frame_idx)
    print("frames: %d, reward: %.3f" % (frame_idx, reward))


def calc_mean_val(batch, net, use_cuda):
    if not batch:
        return 0
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    states, actions, rewards, dones, next_states = batch
    states = torch.tensor(states, dtype=torch.float, device=device)
    mean_val = torch.mean(net(states).max(1)[0])
    return mean_val.item()