import torch
import time
from deep_q_network.lib import dqn_model, wrappers
from deep_q_network.lib.utils import choose_action, HYPERPARAMS

params = HYPERPARAMS['pong']
env = wrappers.make_env(params["env_name"])
net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load("best_weights.pkl"))

s = env.reset()
while True:
    start_time = time.time()
    env.render()
    a = choose_action(env, s, net, 0, use_cuda=False)
    s_, r, done, _ = env.step(a)
    s = s_
    delta = 1/params['fps'] - (time.time()-start_time)
    if delta > 0:
        time.sleep(delta)
    if done:
        break

env.close()
