import os
import sys
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

HIDDEN_SIZE=256
device =  torch.device("cpu")

from typing import Union
Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': torch.nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}

def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)

def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right
    return surrounding

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_observations_mappo(state, agents_index, obs_dim, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state_ = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i, element in enumerate(agents_index):
        # # self head position
        observations[i][:2] = snakes_positions_list[element][0][:]

        # head surroundings
        head_x = snakes_positions_list[element][0][1]
        head_y = snakes_positions_list[element][0][0]

        head_surrounding = get_surrounding(state_, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='softmax'):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, num_agents):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.device = 'cpu'
        self.output_activation = 'softmax'
        self.actor = Actor(obs_dim, act_dim, num_agents, self.output_activation).to(self.device)

    def choose_action(self, obs):
        obs = torch.Tensor([obs]).to(self.device)
        logits = self.actor(obs).cpu().detach().numpy()[0]
        return logits

    def select_action_to_env(self, obs, ctrl_index):
        logits = self.choose_action(obs)
        actions = logits2action(logits)
        action_to_env = to_joint_action(actions, ctrl_index)
        return action_to_env

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename))

def to_joint_action(action, ctrl_index):
    joint_action_ = []
    action_a = action[ctrl_index]
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_

def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = np.array([Categorical(out).sample().item() for out in logits])
    return np.array(actions)

agent_mappo = MAPPOAgent(26, 4, 3)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_200.pth"
agent_mappo.load_model(actor_net)

def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 26
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    observation = get_observations_mappo(obs, indexs, obs_dim, height=board_height, width=board_width)
    actions = agent_mappo.select_action_to_env(observation, indexs.index(o_index-2))
    print(actions)
    return actions
