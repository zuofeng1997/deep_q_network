import numpy as np
import collections
import torch
import torch.nn.functional as F


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
        'Vmin': -10,
        'Vmax': 10,
        'N_ATOMS': 51,
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
        loss_ind = batch_w*(state_action_values-(rewards + gamma * next_state_values))**2
        loss = loss_ind.mean()
        loss.backward()
        optimizer.step()
        return loss_ind+1e-5
    else:
        loss = loss_fn(state_action_values, rewards + gamma * next_state_values)
        loss.backward()
        optimizer.step()


def learn_dist(optimizer, batch, params, net, target_net, use_cuda=True):
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    states, actions, rewards, dones, next_states = batch
    batch_size = len(states)
    states = torch.tensor(states, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float, device=device)

    distr_v = net(states)
    state_action_values = distr_v[range(batch_size), actions]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)


    next_distr_v, next_qvals_v = target_net.both(next_states)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = target_net.apply_softmax(next_distr_v).data.cpu().numpy()
    next_best_distr = next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)
    proj_distr = distr_projection(next_best_distr, rewards, dones, params["Vmin"], params["Vmax"],
                                  params["N_ATOMS"], params["gamma"])
    proj_distr_v = torch.tensor(proj_distr).to(device)
    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = loss_v.sum(dim=1).mean()
    optimizer.zero_grad()
    loss_v.backward()
    optimizer.step()


def learn_rainbow(optimizer, batch, params, net, target_net, batch_w, use_cuda=True):
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    gamma = params["gamma"] * params["gamma"]

    states, actions, rewards, dones, next_states = batch
    batch_size = len(states)
    states = torch.tensor(states, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float, device=device)
    batch_w = torch.tensor(batch_w, device=device)

    distr_v, qvals_v = net.both(torch.cat((states, next_states)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = target_net(next_states)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = target_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()
    dones = dones.astype(np.bool)
    proj_distr = distr_projection(next_best_distr, rewards, dones, params['Vmin'], params['Vmax'], params['N_ATOMS'],
                                  gamma)

    state_action_values = distr_v[range(batch_size), actions]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)

    proj_distr_v = torch.tensor(proj_distr).to(device)
    loss_v_ind = -state_log_sm_v * proj_distr_v
    loss_v = batch_w * loss_v_ind.sum(dim=1)

    optimizer.zero_grad()
    loss_v.mean().backward()
    optimizer.step()
    return loss_v+1e-5


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


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)


        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
    return proj_distr