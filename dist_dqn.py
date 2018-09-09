from deep_q_network.lib import wrappers, dqn_model
from deep_q_network.lib.utils import ExperienceBuffer, HYPERPARAMS, choose_action, logger, learn_dist

import numpy as np
import collections
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

params = HYPERPARAMS["pong"]

DELTA_Z = (params['Vmax']-params['Vmin']) / (params['N_ATOMS'] -1)


class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * params['N_ATOMS'])
        )
        self.register_buffer("supports", torch.arange(params['Vmin'], params['Vmax']+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        test_tensor = self.conv(torch.zeros(1, *shape))
        return int(np.prod(test_tensor.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, params['N_ATOMS'])

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, params['N_ATOMS'])).view(t.size())


def run(double=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    Experience = collections.namedtuple("Experience", field_names=["state", 'action', 'reward', 'done', 'new_state'])
    env = wrappers.make_env(params["env_name"])
    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="basic_dqn")
    buffer = ExperienceBuffer(params["replay_size"])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), params["learning_rate"])
    frame_idx = 0

    while True:
        s = env.reset()
        episode_reward = 0

        while True:
            frame_idx += 1
            epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_frames"])
            a = choose_action(env=env, state=s, net=net.qvals, epsilon=epsilon, use_cuda=args.cuda)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            exp = Experience(s, a, r, done, s_)
            buffer.append(exp)
            if len(buffer) >= params["replay_start_size"]:
                batch = buffer.sample(params["batch_size"])
                learn_dist(optimizer, batch, params, net, target_net, use_cuda=args.cuda)
            if frame_idx % params["target_net_sync"] == 0:
                target_net.load_state_dict(net.state_dict())
            s = s_
            if done:
                break

        logger(writer, frame_idx, episode_reward)

        if episode_reward > params["stop_reward"]:
            print("Solved in %d frames!" % frame_idx)
            torch.save(net.state_dict(), "best_weights.pkl")
            break
    writer.close()


if __name__ == '__main__':
    run()