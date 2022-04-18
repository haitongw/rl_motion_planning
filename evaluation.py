# import numpy as np
# import torch

# import utils
# from envs import make_vec_envs

# def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
#              device):
#     eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
#                               None, eval_log_dir, device, True)

#     vec_norm = utils.get_vec_normalize(eval_envs)
#     if vec_norm is not None:
#         vec_norm.eval()
#         vec_norm.obs_rms = obs_rms

#     eval_episode_rewards = []

#     obs = eval_envs.reset()
#     eval_recurrent_hidden_states = torch.zeros(
#         num_processes, actor_critic.recurrent_hidden_state_size, device=device)
#     eval_masks = torch.zeros(num_processes, 1, device=device)

#     while len(eval_episode_rewards) < 10:
#         with torch.no_grad():
#             _, action, _, eval_recurrent_hidden_states = actor_critic.act(
#                 obs,
#                 eval_recurrent_hidden_states,
#                 eval_masks,
#                 deterministic=True)

#         # Obser reward and next obs
#         obs, _, done, infos = eval_envs.step(action)

#         eval_masks = torch.tensor(
#             [[0.0] if done_ else [1.0] for done_ in done],
#             dtype=torch.float32,
#             device=device)

#         for info in infos:
#             if 'episode' in info.keys():
#                 eval_episode_rewards.append(info['episode']['r'])

#     eval_envs.close()

#     print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
#         len(eval_episode_rewards), np.mean(eval_episode_rewards)))
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

# log_dir = "./log/*.csv"
# dataframe_list = []

# for name in glob.glob(log_dir):
#     dataframe_list.append(pd.read_csv(name))

# num_processes = len(dataframe_list)
# num_episode = len(dataframe_list[0].r)
# return_arr = np.zeros((num_processes,num_episode))

# for idx, df in enumerate(dataframe_list):
#     return_arr[idx,:] = np.array(df.r)

# plt.plot(np.arange(1,num_episode+1), np.mean(return_arr, axis=0))
# plt.show()
log_dir  = "./return_log/point-gremlin-v2-eval.csv"

arr = np.genfromtxt(log_dir)
plt.plot(arr)
plt.show()
# df = pd.read_csv(log_dir)
# print(df)