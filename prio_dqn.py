from deep_q_network.lib import wrappers, dqn_model
from deep_q_network.lib.utils import PrioReplayBuffer, HYPERPARAMS, learn, choose_action, logger

import collections
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


def run(double=False, prio=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    Experience = collections.namedtuple("Experience", field_names=["state", 'action', 'reward', 'done', 'new_state'])
    params = HYPERPARAMS["pong"]
    env = wrappers.make_env(params["env_name"])
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="prio_dqn")
    buffer = PrioReplayBuffer(params["replay_size"], prob_alpha=PRIO_REPLAY_ALPHA)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), params["learning_rate"])
    frame_idx = 0

    while True:
        s = env.reset()
        episode_reward = 0

        while True:
            frame_idx += 1
            epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_frames"])
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            a = choose_action(env=env, state=s, net=net, epsilon=epsilon, use_cuda=args.cuda)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            exp = Experience(s, a, r, done, s_)
            buffer.populate(exp)
            if len(buffer) >= params["replay_start_size"]:
                batch, indices, weights = buffer.sample(params["batch_size"], beta=beta)
                loss_ind = learn(loss_fn, optimizer, batch, params, net, target_net, use_cuda=args.cuda, double=double,
                                 prio=prio, batch_w=weights)
                buffer.update_priorites(indices, loss_ind.data.cpu().numpy())
            if frame_idx % params["target_net_sync"] == 0:
                target_net.load_state_dict(net.state_dict())
            s = s_
            if done:
                break

        logger(writer, frame_idx, episode_reward)
        writer.add_scalar("beta", beta, frame_idx)

        if episode_reward > params["stop_reward"]:
            print("Solved in %d frames!" % frame_idx)
            torch.save(net.state_dict(), "best_weights.pkl")
            break
    writer.close()


if __name__ == '__main__':
    run(prio=True)