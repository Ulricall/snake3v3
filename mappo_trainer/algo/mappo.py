import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device, logits_greedy
from algo.network import Actor, Critic
import torch.nn.functional as F

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class MAPPO:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.clip_epsilon = args.clip_epsilon
        self.output_activation = args.output_activation
        self.lmbda = args.lmbda
        self.epochs = 20

        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
        self.eps = 0.1
        self.decay_speed = 0.95
    
    def choose_action(self, obs):
        if (type(obs) == np.ndarray):
            obs = torch.Tensor([obs]).to(self.device)
        action_distribution = self.actor(obs.detach().clone()).squeeze(0)
        action_distribution = action_distribution.detach().numpy()
        return action_distribution
    
    def take_action(self, state):
        # state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # print(action)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        # Sample a mini-batch of M transitions from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, action_taken_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, 3, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, 3, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, 3, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, 3, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, 3, 1).to(self.device)

        action_taken = torch.Tensor(action_taken_batch).reshape(self.batch_size, 3, 1).to(self.device)
        # action_taken = torch.argmax(action_batch, dim=2).unsqueeze(2).detach()
        # action_dist = torch.distributions.Categorical(action_batch)
        # action_taken = action_dist.sample().unsqueeze(-1)
        # action_taken = self.take_action(state_batch).unsqueeze(-1)
        take_action = torch.as_tensor(action_taken.clone().detach(), dtype=torch.int64)

        # print(reward_batch.shape)
        target = reward_batch + self.gamma * self.critic(next_state_batch, action_batch) * (1 - done_batch)
        target_delta = target - self.critic(state_batch, action_batch)
        advantage = compute_advantage(self.gamma, self.lmbda, target_delta.cpu()).to(self.device)
        pi_old = torch.log(self.actor(state_batch).gather(2, take_action) + 1e-9).detach()

       # print("Epoch start:")
        for _ in range(self.epochs):
            pi_new = torch.log(self.actor(state_batch).gather(2, take_action) + 1e-9)
            # print(pi_new[0])
            ratio = torch.exp(pi_new - pi_old)

            act1 = ratio * advantage
            act2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            loss_actor = torch.mean(-torch.min(act1, act2))
            loss_critic = torch.mean(F.mse_loss(self.critic(state_batch, action_batch), target.detach()))

            # print(loss_actor, loss_critic)
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

        return None, None

    def get_loss(self):
        return self.c_loss, self.a_loss
    
    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')
    
    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic.state_dict(), model_critic_path)