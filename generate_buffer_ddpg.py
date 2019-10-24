import gym
import numpy as np
import torch
import argparse
import os

from BCQ import ddpg
from replay_memory import ReplayMemory


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer_type', default="medio", help='medio|medio-final|optimal|optimal-final')
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument("--noise1", default=0., type=float)  # Probability of selecting random action
    parser.add_argument("--noise2", default=0., type=float)  # Std of Gaussian exploration noise
    args = parser.parse_args()

    name = "%s_DDPG_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))

    print("---------------------------------------")
    print("Settings: " + name)
    print("---------------------------------------")

    if not os.path.exists("./buffers"):
        os.makedirs("./buffers")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # Initialize and load policy
    actor_path = "models/DDPG_actor_{}_{}.pkl".format(args.env_name, args.buffer_type)
    critic_path = "models/DDPG_critic_{}_{}.pkl".format(args.env_name, args.buffer_type)
    policy = ddpg.DDPG(state_dim, action_dim, max_action)
    policy.load(actor_path, critic_path)

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
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                evaluations.append(episode_reward)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Add noise to actions
        if np.random.uniform(0, 1) < args.noise1:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.noise2 != 0:
                action = (action + np.random.normal(0, args.noise2, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        done_bool = 1 if episode_timesteps + 1 == env._max_episode_steps else float(not done)

        # Store data in replay buffer
        memory.push(obs, action, reward, new_obs, done_bool)  # Append transition to memory

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1

    # Save final buffer
    memory.save(name)

    # output mean episode reward and var
    evaluations = np.array(evaluations)
    print("Episode Reward mean: %f variance: %f" % (evaluations.mean(), evaluations.var()))
