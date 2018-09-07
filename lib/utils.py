import numpy as np
import collections
import torch

HYPERPARAMS = {
    'pong': {
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'replay_size':      10000,
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


def learn(loss_fn, optimizer, batch, params, net, target_net, double=False, use_cuda=True):
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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
    loss = loss_fn(state_action_values, rewards + params["gamma"] * params["gamma"] * next_state_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def logger(writer, frame_idx, reward, mean_reward):
    writer.add_scalar("reward", reward, frame_idx)
    writer.add_scalar("mean_reward", mean_reward, frame_idx)
    print("frames: %d, mean_reward: %.3f" % (frame_idx, mean_reward))


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