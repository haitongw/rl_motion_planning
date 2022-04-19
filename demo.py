from model import Policy
from arguments import get_args
import gym
import torch
from env import *
import numpy as np

args = get_args()

device = torch.device("cuda:0" if args.cuda else "cpu")
print(device)
env = gym.make("point-gremlin-v4")
env.seed(42)
np.random.seed(42)
# save_dir = "./trained_models/a2c/point-pillar-v1.pt"
save_dir = "./trained_models/a2c/point-gremlin-v4-net-gru-latest.pt"

actor_critic, _ = torch.load(save_dir)
actor_critic = actor_critic.float()
actor_critic = actor_critic.to(device)
# for param in actor_critic.parameters():
#     print(param.shape)
obs = env.reset()
print(env.action_space.shape)
for i in range(10):
    total_reward = 0
    while True:
        env.render()
        obs_tensor = torch.from_numpy(obs).float().to(device)
        obs_tensor = obs_tensor.unsqueeze(0)
        value, action, _, __ = actor_critic.act(obs_tensor, None, None)
        # act = int(np.squeeze(action.cpu().detach().numpy()))
        # print(action.shape)
        act = action.cpu().detach().numpy()
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(total_reward)
        if done:
            break
    obs = env.reset()

env.close()
