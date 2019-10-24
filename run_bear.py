import argparse
import gym
import numpy as np
import datetime
import os
import itertools
import torch
from bear import BEARQL
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from model import GaussianPolicy


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--buffer_type', default="medio", help='medio|random|optimal')
    parser.add_argument('--genbuffer_algo', default="SAC", help="SAC|DDPG")
    parser.add_argument('--env_name', default="HalfCheetah-v2")
    parser.add_argument('--policy', default="Gaussian", help='Gaussian | Deterministic')
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=1000001)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--init_dual_lambda', type=float, default=1.)
    parser.add_argument('--dual_step_size', type=float, default=0.001)
    parser.add_argument('--cost_epsilon', type=float, default=0.05)
    parser.add_argument('--coefficient_weight', type=float, default=0.6)
    parser.add_argument('--dual_steps', type=int, default=10)
    parser.add_argument('--dirac_policy_num', type=int, default=50)
    parser.add_argument('--m', type=int, default=5, help='sample actor nums when computing mmd')
    parser.add_argument('--n', type=int, default=5, help='sample prior nums when computing mmd')
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    parser.add_argument('--mmd_before_tanh', action="store_true", help='compute mmd distance before tanh or not')
    args = parser.parse_args()

    buffer_name = "%s_%s_%s_%s" % (args.buffer_type, args.genbuffer_algo, args.env_name, str(args.seed))

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    env = gym.make(args.env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Agent
    agent = BEARQL(env.observation_space.shape[0], env.action_space, args)

    # TesnorboardX
    writer = SummaryWriter(
        logdir='runs/{}_BEAR_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                          args.env_name,
                                          args.policy,
                                          )
    )
    # Load buffer
    memory = ReplayMemory(args.replay_size)
    memory.load(buffer_name)

    # load prior
    device = torch.device("cuda" if args.cuda else "cpu")
    prior = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0],
                           args.hidden_size, env.action_space).to(device)
    prior.load_state_dict(torch.load('models/{}_prior_{}_{}.pkl'.format(args.genbuffer_algo, args.env_name, args.buffer_type)))
    print('load prior model: {}_prior_{}_{}.pkl'.format(args.genbuffer_algo, args.env_name, args.buffer_type))
    # prior.load_state_dict(torch.load('models/SAC_actor_{}_{}.pkl'.format(args.env_name, args.buffer_type)))
    # print('load actor model: SAC_actor_{}_{}.pkl'.format(args.env_name, args.buffer_type))

    evaluations = []
    variances = []

    episode_num = 0
    done = True

    for training_iters in range(args.num_steps):

        q_loss, policy_loss, dual_lambda, mmd = agent.train(prior, memory, args.batch_size, args.m, args.n)

        writer.add_scalar('loss/critic', q_loss, training_iters)
        writer.add_scalar('loss/policy', policy_loss, training_iters)
        writer.add_scalar('dual_lambda', dual_lambda, training_iters)
        writer.add_scalar('mmd_distence', mmd, training_iters)

        if training_iters % args.eval_freq == 0:

            eval_res, val_res = evaluate_policy(agent)
            evaluations.append(eval_res)
            variances.append(val_res)
            if not os.path.exists("./results"):
                os.makedirs("./results")
            ""
            np.save("results/reward_%s_%s_BEAR_%s_%s_%s_dual_%s_%s_cost_%s_dirac_policu_num_%s_m_%s_%s" % (
                args.genbuffer_algo, args.buffer_type, args.env_name, str(args.seed), str(args.batch_size),
                str(args.init_dual_lambda), str(args.dual_steps), str(args.cost_epsilon),
                str(args.dirac_policy_num), str(args.m), str(args.mmd_before_tanh),
            ), evaluations)
            np.save("results/variance_%s_%s_BEAR_%s_%s_%s_dual_%s_%s_cost_%s_dirac_policu_num_%s_m_%s_%s" % (
                args.genbuffer_algo, args.buffer_type, args.env_name, str(args.seed), str(args.batch_size),
                str(args.init_dual_lambda), str(args.dual_steps), str(args.cost_epsilon),
                str(args.dirac_policy_num), str(args.m), str(args.mmd_before_tanh),
            ), variances)
            # np.save("results/%s_BEAR_%s_%s" % (args.buffer_type, args.env_name, str(args.seed)), evaluations)

            print("Training iterations: {}".format(str(training_iters)))
            print("loss is {:.3f} | lambda is {:.5f} | mmd is {:.5f} | eval_q is {:.3f} | var is {:.3f}".format(
                q_loss, dual_lambda, mmd, eval_res, val_res
            ))

