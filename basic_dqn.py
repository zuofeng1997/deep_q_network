from deep_q_network.lib import wrappers, dqn_model
from deep_q_network.lib.utils import ExperienceBuffer, HYPERPARAMS, learn, choose_action, logger, calc_mean_val

import numpy as np
import collections
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


def run(double=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    Experience = collections.namedtuple("Experience", field_names=["state", 'action', 'reward', 'done', 'new_state'])
    params = HYPERPARAMS["pong"]
    env = wrappers.make_env(params["env_name"])
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="basic_dqn")
    buffer = ExperienceBuffer(params["replay_size"])
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), params["learning_rate"])
    total_rewards = []
    frame_idx = 0
    batch_eval = None

    while True:
        s = env.reset()
        episode_reward = 0

        while True:
            frame_idx += 1
            epsilon = max(params["epsilon_final"], params["epsilon_start"] - frame_idx / params["epsilon_frames"])
            a = choose_action(env=env, state=s, net=net, epsilon=epsilon, use_cuda=args.cuda)
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
        total_rewards.append(episode_reward)
        mean_reward = np.mean(total_rewards[-100:])
        logger(writer, frame_idx, epsilon, episode_reward, mean_reward)

        if len(buffer) == params["replay_size"]:
            if not batch_eval:
                batch_eval = buffer.sample(params["state_eval_num"])
        mean_val = calc_mean_val(batch_eval, net=net, use_cuda=args.cuda)

        writer.add_scalar("mean_val", mean_val, frame_idx)

        if mean_reward > params["stop_reward"]:
            print("Solved in %d frames!" % frame_idx)
            torch.save(net.state_dict(), "best_weights.pkl")
            break
    writer.close()


if __name__ == '__main__':
    run()