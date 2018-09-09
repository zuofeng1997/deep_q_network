from deep_q_network.lib import wrappers, dqn_model
from deep_q_network.lib.utils import PrioReplayBuffer, HYPERPARAMS, learn_rainbow, choose_action, logger

import numpy as np
import collections
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

params = HYPERPARAMS["pong"]
DELTA_Z = (params['Vmax']-params['Vmin']) / (params['N_ATOMS'] -1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_val = nn.Sequential(
            dqn_model.NoisyFactorizedLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyFactorizedLinear(512, params['N_ATOMS'])
        )
        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            dqn_model.NoisyLinear(512, n_actions * params['N_ATOMS'])
        )
        self.register_buffer("supports", torch.arange(params['Vmin'], params['Vmax'] + DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, params['N_ATOMS'])
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, params['N_ATOMS'])
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

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


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    Experience = collections.namedtuple("Experience", field_names=["state", 'action', 'reward', 'done', 'new_state'])
    params = HYPERPARAMS["pong"]
    env = wrappers.make_env(params["env_name"])
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="prio_dqn")
    buffer = PrioReplayBuffer(params["replay_size"], prob_alpha=PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), params["learning_rate"])
    frame_idx = 0
    epsilon = 0
    while True:
        s = env.reset()
        episode_reward = 0

        while True:
            frame_idx += 1
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            a = choose_action(env=env, state=s, net=net.qvals, epsilon=epsilon, use_cuda=args.cuda)
            s_, r, done, _ = env.step(a)
            episode_reward += r

            if done:
                exp = Experience(s, a, r, done, s_)
                buffer.populate(exp)
            else:
                a_ = choose_action(env=env, state=s_, net=net.qvals, epsilon=epsilon, use_cuda=args.cuda)
                s__, r_, done_, __ = env.step(a_)
                episode_reward += r_
                two_steps_r = r + params["gamma"]*r_
                exp = Experience(s, a, two_steps_r, done_, s__)
                buffer.populate(exp)

            if len(buffer) >= params["replay_start_size"]:
                batch, indices, weights = buffer.sample(params["batch_size"], beta=beta)
                loss_ind = learn_rainbow(optimizer, batch, params, net, target_net, batch_w=weights)
                buffer.update_priorites(indices, loss_ind.data.cpu().numpy())
            if frame_idx % params["target_net_sync"] == 0:
                target_net.load_state_dict(net.state_dict())

            if done or done_:
                break

            s = s__

        logger(writer, frame_idx, episode_reward)
        writer.add_scalar("beta", beta, frame_idx)

        if episode_reward > params["stop_reward"]:
            print("Solved in %d frames!" % frame_idx)
            torch.save(net.state_dict(), "best_weights.pkl")
            break
    writer.close()


if __name__ == '__main__':
    run()