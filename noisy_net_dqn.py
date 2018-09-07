from deep_q_network.lib import wrappers, dqn_model
from deep_q_network.lib.utils import ExperienceBuffer, HYPERPARAMS, learn, choose_action, logger

import numpy as np
import collections
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            dqn_model.NoisyFactorizedLinear(conv_out_size, 512),
            dqn_model.NoisyFactorizedLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        test_tensor = self.conv(torch.zeros(1, *shape))
        return int(np.prod(test_tensor.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


def run(double=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    Experience = collections.namedtuple("Experience", field_names=["state", 'action', 'reward', 'done', 'new_state'])
    params = HYPERPARAMS["pong"]
    env = wrappers.make_env(params["env_name"])
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="noisy_net_dqn")
    buffer = ExperienceBuffer(params["replay_size"])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), params["learning_rate"])
    frame_idx = 0

    while True:
        s = env.reset()
        episode_reward = 0

        while True:
            frame_idx += 1
            # epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_frames"])
            a = choose_action(env=env, state=s, net=net, epsilon=0, use_cuda=args.cuda)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            exp = Experience(s, a, r, done, s_)
            buffer.append(exp)
            if len(buffer) >= params["replay_start_size"]:
                batch = buffer.sample(params["batch_size"])
                learn(loss_fn, optimizer, batch, params, net, target_net, use_cuda=args.cuda, double=double)
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