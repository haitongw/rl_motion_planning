from model import Policy
from arguments import get_args
import gym
import torch
from env import *
import numpy as np

args = get_args()

device = torch.device("cuda:0" if args.cuda else "cpu")
print(device)
env = gym.make("point-pillar-v1")

save_dir = "./trained_models/a2c/point-pillar-v1.pt"
# save_dir = "./trained_models/a2c/point-simple-v0.pt"

actor_critic, _ = torch.load(save_dir)
actor_critic = actor_critic.float()
actor_critic = actor_critic.to(device)

obs = env.reset()

for i in range(10):
    while True:
        env.render()
        obs_tensor = torch.from_numpy(obs).float().to(device)
        value, action, _, __ = actor_critic.act(obs_tensor, None, None)
        # act = int(np.squeeze(action.cpu().detach().numpy()))
        obs, reward, done, info = env.step(action.cpu().detach().numpy())
        print(reward)
        if done:
            break
    obs = env.reset()

env.close()
