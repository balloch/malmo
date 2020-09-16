import os
import malmoenv
import argparse
from pathlib import Path
import time
from PIL import Image
from collections import deque

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from guided import *

def main():
    args = get_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    env = malmoenv.make()

    env.init(xml, args.port,
             server=args.server,
             server2=args.server2, port2=args.port2,
             role=args.role,
             exp_uid=args.experimentUniqueId,
             episode=args.episode, 
             resync=args.resync,
             reshape=True,)

    #obs_shape = (env.observation_space.shape[2],env.observation_space.shape[0],env.observation_space.shape[1])
    obs_shape = env.observation_space.shape
    #    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
    #        env = TransposeImage(env, op=[2, 0, 1])

    if args.guided:
        pass
    else:
        actor_critic = Policy(
            obs_shape,
            env.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)
        

        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = env.reset()
    obs = torch.from_numpy(obs).float().to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

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
            obs, reward, done, infos = env.step(action)
            if reward is None:
                continue #reward = 0.0
            obs = torch.from_numpy(obs).float().to(device)
           # for info in infos:
           #     if 'episode' in info.keys():
           #     episode_rewards.append(info['episode']['r'])
            if done or step > args.episodemaxsteps:
                episode_rewards.append(reward)
                done = False
                obs = env.reset()
                obs = torch.from_numpy(obs).float().to(device)
                break
            # If done then clean the history of observations.
#             masks = torch.FloatTensor(
#                 [[0.0] if done_ else [1.0] for done_ in done])
#             bad_masks = torch.FloatTensor(
#                 [[0.0] if 'bad_transition' in info.keys() else [1.0]
#                  for info in infos])
            # Hardcode for testing
            masks = torch.FloatTensor([[1.0]]) #always not done
            bad_masks = torch.FloatTensor([[1.0]]) #always good transitions
            
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, torch.FloatTensor([reward]), masks, bad_masks)
            #print(reward)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
     
        # obs = env.reset()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        print("{} Steps, Value Loss: {}, Action Loss: {}".format(total_num_steps, value_loss, action_loss))
        rollouts.after_update()

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


        steps = 0
        done = False
       # while not done and (args.episodemaxsteps <= 0 or steps < args.episodemaxsteps):
       #     action = env.action_space.sample()

       #     obs, reward, done, info = env.step(action)
       #     obs = torch.from_numpy(obs).float().to(device) #unsqueeze?
       #     #reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
       #     steps += 1
       #     print("reward: " + str(reward))
       #     # print("done: " + str(done))
       #     print("obs: " + str(obs))
       #     # print("info" + info)
       #     if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
       #         d, h, w = env.observation_space.shape
       #         img = Image.fromarray(obs.reshape(h, w, d))
       #         img.save('image' + str(args.role) + '_' + str(steps) + '.png')

        time.sleep(.05)

    env.close()


if __name__ == '__main__':
    main()
