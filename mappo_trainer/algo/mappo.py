import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network import Actor, Critic

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

        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        hard_update(self.actor, self.actor_target)
        hard_update(self.critic, self.critic_target)

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
        self.eps = 1
        self.decay_speed = 0.95
        self.c_loss = None
        self.a_loss = None
    
    def choose_action(self, obs):
        if (type(obs) == np.ndarray):
            obs = torch.Tensor([obs]).to(self.device)
        action_distribution = self.actor(obs.detach().clone())

        return action_distribution.squeeze(0)

    def update(self):
        with torch.autograd.set_detect_anomaly(True):
            if len(self.replay_buffer) < self.batch_size:
                return None, None
            # Sample a mini-batch of M transitions from memory
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_probs_batch = self.replay_buffer.get_batches()

            state_batch = torch.Tensor(state_batch).reshape(self.batch_size, 3, -1).to(self.device)
            action_batch = torch.Tensor(action_batch).reshape(self.batch_size, 3, -1).to(self.device)
            reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, 3, -1).to(self.device)
            next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, 3, -1).to(self.device)
            done_batch = torch.Tensor(done_batch).reshape(self.batch_size, 3, 1).to(self.device)
            log_probs_batch = torch.Tensor(log_probs_batch).reshape(self.batch_size, 3, -1).to(self.device)

            with torch.no_grad():
                target_next_actions = self.actor_target(next_state_batch)
                next_value = self.critic_target(next_state_batch, target_next_actions)
                target_value = reward_batch[:,:,0].unsqueeze(2) + self.gamma * next_value * (1 - done_batch)

            # Update the critic
            # print(state_batch.shape, action_batch.shape)
            value = self.critic(state_batch, action_batch)
            loss_critic = torch.nn.MSELoss()(target_value, value)

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            clip_grad_norm_(self.critic.parameters(), 1)
            self.critic_optimizer.step()

            new_logits = self.actor(state_batch.detach().clone())
            new_actions = torch.argmax(new_logits, dim=2)
            new_log_probs = torch.log(torch.stack([new_logits[_].gather(1, new_actions[_].unsqueeze(0)) for _ in range(self.batch_size)], dim=0)).unsqueeze(0)
            ratio = torch.exp(new_log_probs - log_probs_batch)
            advantage = target_value.detach() - value.detach()
            loss_actor_1 = ratio * advantage
            loss_actor_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            loss_actor = -torch.min(loss_actor_1, loss_actor_2).mean()

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()

            self.c_loss = loss_critic.item()
            self.a_loss = loss_actor.item()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)

            return self.c_loss, self.a_loss

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