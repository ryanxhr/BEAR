import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class BEARQL(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), weight_decay=1e-2)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=1e-4)

        self.policy_target = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        hard_update(self.policy_target, self.policy)

        self.dual_lambda = args.init_dual_lambda
        self.dual_step_size = args.dual_step_size
        self.cost_epsilon = args.cost_epsilon
        self.coefficient_weight = args.coefficient_weight
        self.dual_steps = args.dual_steps
        self.dirac_policy_num = args.dirac_policy_num
        self.m = args.m
        self.n = args.n
        self.mmd_before_tanh = args.mmd_before_tanh

    # used in evaluation
    def select_action(self, state):
        # sample multiple policies and perform a greedy maximization of Q over these policies
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(self.dirac_policy_num, 1).to(self.device)
            # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            x_t, action, _, mean = self.policy.sample(state)
            # q1, q2 = self.critic(state, action)
            q1, q2, q3 = self.critic(state, action)
            ind = (q1+q2+q3).max(0)[1]
        return action[ind].cpu().data.numpy().flatten()
        # return action.cpu().data.numpy().flatten()

    # MMD functions
    def compute_gau_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-(tiled_x - tiled_y).pow(2).sum(dim=3) / (2 * sigma))

    # MMD functions
    def compute_lap_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-torch.abs(tiled_x - tiled_y).sum(dim=3) / sigma)

    def compute_mmd(self, x, y, kernel='lap'):
        if kernel == 'gau':
            x_kernel = self.compute_gau_kernel(x, x, 20)
            y_kernel = self.compute_gau_kernel(y, y, 20)
            xy_kernel = self.compute_gau_kernel(x, y, 20)
        else:
            x_kernel = self.compute_lap_kernel(x, x, 10)
            y_kernel = self.compute_lap_kernel(y, y, 10)
            xy_kernel = self.compute_lap_kernel(x, y, 10)
        square_mmd = x_kernel.mean((1, 2)) + y_kernel.mean((1, 2)) - 2 * xy_kernel.mean((1, 2))
        return square_mmd

    def train(self, prior, memory, batch_size, m=4, n=4):
        # Sample replay buffer / batch
        state_np, action_np, reward_np, next_state_np, mask_np = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_np).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_np).to(self.device)
        action_batch = torch.FloatTensor(action_np).to(self.device)
        reward_batch = torch.FloatTensor(reward_np).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_np).to(self.device).unsqueeze(1)

        # Critic Training
        with torch.no_grad():
            # Duplicate state 10 times
            next_state_rep = torch.FloatTensor(np.repeat(next_state_np, self.dirac_policy_num, axis=0)).to(self.device)

            # Soft Clipped Double Q-learning
            _, next_state_action, _, _ = self.policy_target.sample(next_state_rep)
            target_Q1, target_Q2, target_Q3 = self.critic_target(next_state_rep, next_state_action)
            target_cat = torch.cat([target_Q1, target_Q2, target_Q3], 1)
            target_Q = 0.75 * target_cat.min(1)[0] + 0.25 * target_cat.max(1)[0]
            target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)
            # target_Q1, target_Q2 = self.critic_target(next_state_rep, next_state_action)
            # target_Q = 0.75 * torch.min(target_Q1, target_Q2) + 0.25 * torch.max(target_Q1, target_Q2)
            # target_Q = target_Q.view(batch_size, -1).max(1)[0].view(-1, 1)

            next_q_value = reward_batch + mask_batch * self.gamma * target_Q

        qf1, qf2, qf3 = self.critic(state_batch, action_batch)  # ensemble of k Q-functions
        q_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value) + F.mse_loss(qf3, next_q_value)
        # qf1, qf2 = self.critic(state_batch, action_batch)  # ensemble of k Q-functions
        # q_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Train Actor
        with torch.no_grad():
            state_rep_m = torch.FloatTensor(np.repeat(state_np, m, axis=0)).to(self.device)
            state_rep_n = torch.FloatTensor(np.repeat(state_np, n, axis=0)).to(self.device)
            prior_x_t, prior_a, _, _ = prior.sample(state_rep_n)
            prior_a = prior_a.view(batch_size, n, -1)
            prior_x_t = prior_x_t.view(batch_size, n, -1)

        for s in range(self.dual_steps):
            x_t_rep, a_rep, _, _ = self.policy.sample(state_rep_m)
            if self.mmd_before_tanh:
                x_t_rep = x_t_rep.view(batch_size, m, -1)
                mmd_dist = self.compute_mmd(prior_x_t, x_t_rep)
            else:
                a_rep = a_rep.view(batch_size, m, -1)
                mmd_dist = self.compute_mmd(prior_a, a_rep)

            _, pi, _, _ = self.policy.sample(state_batch)
            qf1_pi, qf2_pi, qf3_pi = self.critic(state_batch, pi)
            qf_cat = torch.cat([qf1_pi, qf2_pi, qf3_pi], 1)
            qf_mean = qf_cat.mean(1)
            qf_var = qf_cat.var(1)
            min_qf_pi = qf_mean - self.coefficient_weight * qf_var.sqrt()  # used in BEAR
            # qf1_pi, qf2_pi = self.critic(state_batch, pi)
            # min_qf_pi = qf1_pi

            policy_loss = -(min_qf_pi - self.dual_lambda * (mmd_dist - self.cost_epsilon)).mean()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Dual Lambda Training
            self.dual_gradients = mmd_dist.mean().item() - self.cost_epsilon
            self.dual_lambda += self.dual_step_size * self.dual_gradients
            self.dual_lambda = np.clip(self.dual_lambda, np.power(np.e, -5), np.power(np.e, 10))

        # Update Target Networks
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)

        return q_loss.item(), policy_loss.item(), self.dual_lambda, mmd_dist.mean().item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/BEAR_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/BEAR_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))













