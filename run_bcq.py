import gym
import numpy as np
import torch
import argparse
import os

from replay_memory import ReplayMemory
from BCQ import bcq


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        reward = 0.
        while not done:
            action = policy.select_action(np.array(obs))
            obs, r, done, _ = env.step(action)
            reward += r
        avg_reward.append(reward)

    avg_reward = np.array(avg_reward)

    # print("---------------------------------------")
    # print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    # print("---------------------------------------")
    return avg_reward.mean(), avg_reward.var()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=123456, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--buffer_type", default="medio")  # Prepends name to filename.
    parser.add_argument('--genbuffer_algo', default="SAC", help="SAC|DDPG")
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--ac_hidden1', type=int, default=400)
    parser.add_argument('--ac_hidden2', type=int, default=300)
    parser.add_argument('--vae_hidden', type=int, default=750)
    args = parser.parse_args()

    buffer_name = "%s_%s_%s_%s" % (args.buffer_type, args.genbuffer_algo, args.env_name, str(args.seed))

    print("---------------------------------------")
    print("Settings: " + buffer_name)
    print("---------------------------------------")

    env = gym.make(args.env_name)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = bcq.BCQ(state_dim, action_dim, max_action, (args.ac_hidden1, args.ac_hidden2), args.vae_hidden)

    # Load buffer
    memory = ReplayMemory(args.replay_size)
    memory.load(buffer_name)

    evaluations = []
    variances = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < args.max_timesteps:
        q_loss, policy_loss = policy.train(memory, iterations=int(args.eval_freq))

        eval_res, val_res = evaluate_policy(policy)
        evaluations.append(eval_res)
        variances.append(val_res)

        if not os.path.exists("./results"):
            os.makedirs("./results")
        np.save("results/reward_%s_BCQ_%s_%s" % (args.buffer_type, args.env_name, str(args.seed)), evaluations)
        np.save("results/variance_%s_BCQ_%s_%s" % (args.buffer_type, args.env_name, str(args.seed)), variances)

        training_iters += args.eval_freq
        print("Training iterations: " + str(training_iters))
        print("loss is {:.3f} | eval_q is {:.3f} | var is {:.3f}".format(
            q_loss, eval_res, val_res
        ))
