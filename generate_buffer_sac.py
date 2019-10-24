import gym
import numpy as np
import torch
import argparse
import os

from sac import SAC
from replay_memory import ReplayMemory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--buffer_type', default="medio", help='medio|random|optimal')
    parser.add_argument('--env_name', default="HalfCheetah-v2")
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=123456)
    # default sac argparse
    parser.add_argument('--policy', default="Gaussian")
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=1000001)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--cuda', action="store_true")
    args = parser.parse_args()

    name = "%s_SAC_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + name)
    print("---------------------------------------")

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Initialize and load policy
    actor_path = "models/SAC_actor_{}_{}.pkl".format(args.env_name, args.buffer_type)
    critic_path = "models/SAC_critic_{}_{}.pkl".format(args.env_name, args.buffer_type)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(actor_path, critic_path)

    # Initialize buffer
    memory = ReplayMemory(args.replay_size)

    evaluations = []

    total_timesteps = 0
    episode_num = 0
    done = True

    while total_timesteps < args.replay_size:

        if done:

            if total_timesteps != 0:
                print("Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                    total_timesteps, episode_num, episode_steps, episode_reward))
                evaluations.append(episode_reward)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_num += 1

        # action
        action = agent.select_action(np.array(obs))

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(obs, action, reward, new_obs, mask)  # Append transition to memory

        obs = new_obs

        episode_steps += 1
        total_timesteps += 1

    # Save final buffer
    memory.save(name)

    # output mean episode reward and var
    evaluations = np.array(evaluations)
    print("Episode Reward mean: %f variance: %f" % (evaluations.mean(), evaluations.var()))
