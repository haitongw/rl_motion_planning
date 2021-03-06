"""
@misc{pytorchrl,
  author = {Kostrikov, Ilya},
  title = {PyTorch Implementations of Reinforcement Learning Algorithms},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail}},
}
"""

import copy
import os
import time
from collections import deque
import csv

import gym
import safety_gym
import env

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from algo import A2C_ACKTR
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
# from evaluation import evaluate

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)  # replace ~ with home directory
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)    # CPU
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(device)

    # multiprocessing, record episode,....
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.model is not None:
        print("load model: ",args.model)
        model_dir = "./trained_models/a2c/" + args.model
        actor_critic, _ = torch.load(model_dir)
    else:
        # neural networks
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy,'hidden_size': 256,'device':device})
    actor_critic.to(device)

    # neural nets update
    agent = A2C_ACKTR(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    best_mean_reward = -np.inf
    return_storage = np.zeros(args.num_processes)
    return_storeage_cnt = 0

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print(f"There will be {num_updates} updates.")
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            return_storage += np.squeeze(reward.numpy())
            return_storeage_cnt += 1
            if return_storeage_cnt >= args.eval_interval:
                # print(f"average return {return_storage}")
                log_dir = "./return_log/" + args.env_name + "-eval.csv"
                with open(log_dir,'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([np.mean(return_storage)])
                return_storage = np.zeros(args.num_processes)
                return_storeage_cnt = 0

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            # masks are used to indicate whether it's a true terminal state
            # or time limit end state
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            
            if len(episode_rewards)>1 and np.mean(episode_rewards) > best_mean_reward:
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + "-net-gru-eval-best.pt"))

                best_mean_reward = np.mean(episode_rewards)

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + "-net-gru-eval-latest.pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

if __name__ == "__main__":
    main()
