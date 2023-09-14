import numpy as np
import torch
# import gym
import argparse
import os
# import d4rl
from tqdm import trange
# from coolname import generate_slug
import time
# import json
from log import Logger

import utils
from utils import VideoRecorder
import IQL

import wandb

from torch.utils.data import DataLoader, TensorDataset
from import_off_data import ImportData
from buffer import ReplayBuffer
from off_env import Env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


# def eval_policy(args, iter, video: VideoRecorder, logger: Logger, policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
#     eval_env = gym.make(env_name)
#     eval_env.seed(seed + seed_offset)

#     lengths = []
#     returns = []
#     avg_reward = 0.
#     for _ in range(eval_episodes):
#         video.init(enabled=(args.save_video and _ == 0))
#         state, done = eval_env.reset(), False
#         video.record(eval_env)
#         steps = 0
#         episode_return = 0
#         while not done:
#             state = (np.array(state).reshape(1, -1) - mean)/std
#             action = policy.select_action(state)
#             state, reward, done, _ = eval_env.step(action)
#             video.record(eval_env)
#             avg_reward += reward
#             episode_return += reward
#             steps += 1
#         lengths.append(steps)
#         returns.append(episode_return)
#         video.save(f'eval_s{iter}_r{str(episode_return)}.mp4')

#     avg_reward /= eval_episodes
#     d4rl_score = eval_env.get_normalized_score(avg_reward)

#     logger.log('eval/lengths_mean', np.mean(lengths), iter)
#     logger.log('eval/lengths_std', np.std(lengths), iter)
#     logger.log('eval/returns_mean', np.mean(returns), iter)
#     logger.log('eval/returns_std', np.std(returns), iter)
#     logger.log('eval/d4rl_score', d4rl_score, iter)

#     print("---------------------------------------")
#     print(f"Evaluation over {eval_episodes} episodes: {d4rl_score:.3f}")
#     print("---------------------------------------")
#     return d4rl_score

def get_config():
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="IQL")               # Policy name
    parser.add_argument("--env", default="outdoor off RL")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
    parser.add_argument("--save_model", action="store_true", default=False)        # Save model and optimizer parameters
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument("--normalize", default=False, action='store_true')
    # IQL
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--temperature", default=3.0, type=float)
    parser.add_argument("--expectile", default=0.7, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    # Work dir
    parser.add_argument('--work_dir', default='/home/kasun/offlineRL/IQL/iql-pytorch', type=str)
    args = parser.parse_args()
    # args.cooldir = generate_slug(2)
    return args

if __name__ == "__main__":

    config = get_config()

    # # Build work dir
    # base_dir = 'runs'
    # utils.make_dir(base_dir)
    # base_dir = os.path.join(base_dir, args.work_dir)
    # utils.make_dir(base_dir)
    # args.work_dir = os.path.join(base_dir, args.env)
    # utils.make_dir(args.work_dir)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d-%H:%M", ts)
    exp_name = str(config.env) + '-' + ts + '-bs' + str(config.batch_size) + '-s' + str(config.seed)
    if config.policy == 'IQL':
        exp_name += '-t' + str(config.temperature) + '-e' + str(config.expectile)
    else:
        raise NotImplementedError
    # exp_name += '-' + config.cooldir
    config.work_dir = config.work_dir #+ '/' + exp_name
    # utils.make_dir(config.work_dir)

    config.model_dir = os.path.join(config.work_dir, 'model')
    # utils.make_dir(config.model_dir)
    # config.video_dir = os.path.join(config.work_dir, 'video')
    # utils.make_dir(config.video_dir)

    # with open(os.path.join(config.work_dir, 'config.json'), 'w') as f:
    #     json.dump(vars(config), f, sort_keys=True, indent=4)

    # utils.snapshot_src('.', os.path.join(config.work_dir, 'src'), '.gitignore')

    # print("---------------------------------------")
    # print(f"Policy: {config.policy}, Env: {config.env}, Seed: {config.seed}")
    # print("---------------------------------------")

    env = Env(True,'/home/kasun/offlineRL/dataset') #gym.make(config.env)
    train_dataset = ImportData('/media/kasun/Media/offRL/dataset/')

    dataloader  = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

    # Set seeds
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    state_dim = (3,40,40) #env.observation_space.shape[0]
    action_dim = (3,1) #env.action_space.shape[0]
    max_action = 0.9 #float(env.action_space.high[0])
    config.max_timesteps = 250

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        # IQL
        "discount": config.discount,
        "tau": config.tau,
        "temperature": config.temperature,
        "expectile": config.expectile,
    }

    # Initialize policy
    if config.policy == 'IQL':
        policy = IQL.IQL(**kwargs)
    else:
        raise NotImplementedError

    buffer_size=int(1e6)

    # replay_buffer = ReplayBuffer(buffer_size, config.batch_size, device) #utils.ReplayBuffer(state_dim, action_dim)
    # replay_buffer.add(dataloader)

    # replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    # if 'antmaze' in x2.env:
    #     # Center reward for Ant-Maze
    #     # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    #     replay_buffer.reward = replay_buffer.reward - 1.0
    # if x2.normalize:
    #     mean, std = replay_buffer.normalize_states()
    # else:
    #     mean, std = 0, 1

    logger = Logger(config.work_dir, use_tb=True)
    # video = VideoRecorder(dir_name=x2.video_dir)

    
    with wandb.init(project="Outdoor-IQL-offline", name="Spot outdoor", config=config):

        for t in trange(int(config.max_timesteps)):

            for batch_idx, experience in enumerate(dataloader):
                # states, actions, rewards, next_states, dones = experience
                # states_1,states_2, actions, rewards, next_states_1, next_states_2,dones = experience

                actor_loss, adv, critic_loss, q1,q2 = policy.train(experience, config.batch_size, device,logger=logger)

            wandb.log({
            "actor_loss": actor_loss,
            "adv": adv,
            "critic_loss": critic_loss,
            "q1": q1,
            "q2": q2,
            "Episode": t})

            # # Evaluate episode
            # if (t + 1) % x2.eval_freq == 0:
            #     eval_episodes = 100 if t+1 == int(x2.max_timesteps) else x2.eval_episodes
            #     d4rl_score = eval_policy(args, t+1, video, logger, policy, args.env,
            #                              args.seed, mean, std, eval_episodes=eval_episodes)
            #     if args.save_model:
            #         policy.save(args.model_dir)
            if t % 40 == 0:
                torch.save(policy.state_dict(), os.path.join('./trained_models/','iql_sep13k_'+str(t)+'.pkl'))
                # policy.save(args.model_dir)

        # logger._sw.close()
