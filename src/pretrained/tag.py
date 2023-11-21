import gym
from gym.spaces import Tuple
from pretrained.ddpg import DDPG
import torch
import os

class FrozenTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:-1]

    def step(self, action):
        random_action = 0
        action = tuple(action) + (random_action,)
        obs, rew, done, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info

class RandomTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:-1]

    def step(self, action):
        random_action = self.pt_action_space.sample()
        action = tuple(action) + (random_action,)
        obs, rew, done, info = super().step(action)
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info


class PretrainedTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

        self.prey = DDPG(14, 5, 50, 128, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), 'prey_params.pt')
        save_dict = torch.load(param_path)
        self.prey.load_params(save_dict['agent_params'][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = obs[-1]
        return obs[:-1]

    def step(self, action):
        prey_action = self.prey.step(self.last_prey_obs)
        action = tuple(action) + (prey_action,)
        obs, rew, done, info = super().step(action)
        self.last_prey_obs = obs[-1]
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info


class HeuristicTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

        self.prey = DDPG(14, 5, 50, 128, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), 'prey_params.pt')
        save_dict = torch.load(param_path)
        self.prey.load_params(save_dict['agent_params'][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = obs[-1]
        return obs[:-1]

    def step(self, action):
        self_pos = self.last_prey_obs[2:4]
        predator_pos_lst = [self.last_prey_obs[8:10], self.last_prey_obs[10:12], self.last_prey_obs[12:14]]
        dist_lst = [((pos - self_pos) ** 2).sum() for pos in predator_pos_lst]
        min_dist = min(dist_lst)
        closet_predator_pos = predator_pos_lst[dist_lst.index(min_dist)]
        x, y = closet_predator_pos
        if x >= 0 and abs(x) >= abs(y):
            prey_action = 2
        elif x < 0 and abs(x) >= abs(y):
            prey_action = 1
        elif y >= 0 and abs(y) >= abs(x):
            prey_action = 4
        elif y < 0 and abs(y) >= abs(x):
            prey_action = 3
        else:
            print(self.last_prey_obs)
            print(predator_pos_lst)
            print(dist_lst)
            print(min_dist)
            print(closet_predator_pos)
            print(x,y)
            assert 0

        action = tuple(action) + (prey_action,)
        obs, rew, done, info = super().step(action)
        self.last_prey_obs = obs[-1]
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info


class BullTag(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pt_action_space = self.action_space[-1]
        self.pt_observation_space = self.observation_space[-1]
        self.action_space = Tuple(self.action_space[:-1])
        self.observation_space = Tuple(self.observation_space[:-1])
        self.n_agents = 3

        self.prey = DDPG(14, 5, 50, 128, 0.01)
        param_path = os.path.join(os.path.dirname(__file__), 'prey_params.pt')
        save_dict = torch.load(param_path)
        self.prey.load_params(save_dict['agent_params'][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None


    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        self.last_prey_obs = obs[-1]
        return obs[:-1]

    def step(self, action):
        self_pos = self.last_prey_obs[2:4]
        predator_pos_lst = [self.last_prey_obs[8:10], self.last_prey_obs[10:12], self.last_prey_obs[12:14]]
        dist_lst = [((pos - self_pos) ** 2).sum() for pos in predator_pos_lst]
        min_dist = min(dist_lst)
        closet_predator_pos = predator_pos_lst[dist_lst.index(min_dist)]
        x, y = closet_predator_pos
        if x >= 0 and abs(x) >= abs(y):
            prey_action = 1
        elif x < 0 and abs(x) >= abs(y):
            prey_action = 2
        elif y >= 0 and abs(y) >= abs(x):
            prey_action = 3
        elif y < 0 and abs(y) >= abs(x):
            prey_action = 4
        else:
            assert 0

        action = tuple(action) + (prey_action,)
        obs, rew, done, info = super().step(action)
        self.last_prey_obs = obs[-1]
        obs = obs[:-1]
        rew = rew[:-1]
        done = done[:-1]
        return obs, rew, done, info