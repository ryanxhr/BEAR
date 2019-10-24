import torch
import argparse
import numpy as np
import gym
import os
from torch.optim import Adam
from model import GaussianPolicy
from replay_memory import ReplayMemory


class fit_dist(object):
    def __init__(self, num_inputs, action_space, args):
        self.device = torch.device("cpu")
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        self.genbuffer_algo = args.genbuffer_algo

    def train(self, memory, batch_size):
        # Sample replay buffer / batch
        state_np, action_np, reward_np, next_state_np, mask_np = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_np).to(self.device)
        action_batch = torch.FloatTensor(action_np).to(self.device)

        log_prob = self.policy.lod_prob(state_batch, action_batch)
        loss = -log_prob.mean()

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()

        return loss.item()

    # Save model parameters
    def save_model(self, buffer_type, genbuffer_algo, env_name, suffix="", actor_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/{}_prior_{}_{}.{}".format(genbuffer_algo, env_name, buffer_type, suffix)
        print('Saving models to {}'.format(actor_path))
        torch.save(self.policy.state_dict(), actor_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--buffer_type', default="medio", help='medio|random|optimal|imitation')
    parser.add_argument('--genbuffer_algo', default="SAC", help="SAC|DDPG")
    parser.add_argument('--env_name', default="HalfCheetah-v2")
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--hidden_size', type=int, default=256)
    args = parser.parse_args()

    name = "%s_%s_%s_%s" % (args.buffer_type, args.genbuffer_algo, args.env_name, str(args.seed))

    env = gym.make(args.env_name)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    agent = fit_dist(env.observation_space.shape[0], env.action_space, args)

    memory = ReplayMemory(args.replay_size)
    memory.load(name)
    for i in range(args.epoch):
        loss = agent.train(memory, args.batch_size)
        print("iteration: {}, loss is {}".format(i, loss))

    # save model
    agent.save_model(args.buffer_type, args.genbuffer_algo, args.env_name, suffix='pkl')






