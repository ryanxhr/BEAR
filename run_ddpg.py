import numpy as np
import torch
import gym
import argparse
import os

from replay_memory import ReplayMemory
from BCQ import ddpg

# Shortened version of code originally found at https://github.com/sfujim/TD3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--limit', type=float, default=np.inf)
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--start_timesteps", default=1e3, type=int)  # Run times of purely random policy
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument('--buffer_type', default="imitation")
    parser.add_argument('--replay_size', type=int, default=1000000)
    args = parser.parse_args()

    file_name = "%s_%s" % (args.env_name, args.buffer_type)
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy and buffer
    policy = ddpg.DDPG(state_dim, action_dim, max_action)
    memory = ReplayMemory(args.replay_size)

    total_timesteps = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {:.1f}"
                      .format(total_timesteps, episode_num, episode_timesteps, episode_reward))
                policy.train(memory, episode_timesteps)
                if episode_reward > args.limit:
                    break

            # Save policy
            if total_timesteps % 1e5 == 0:
                policy.save(file_name, directory="./models")

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 1 if episode_timesteps + 1 == env._max_episode_steps else float(not done)
        episode_reward += reward

        # Store data in replay buffer
        memory.push(obs, action, reward, new_obs, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1

    # Save final policy
    policy.save("%s" % (file_name), directory="./models")

    # Save buffer
    # if not os.path.exists("./buffers"):
    #     os.makedirs("./buffers")
    # memory.save(filename=file_name)
