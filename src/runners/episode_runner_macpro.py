from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th
from copy import deepcopy

class EpisodeRunnerMACPro:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, pred_task=None, mode=None):

        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.reset_epsilon:
                t_eps = self.t_env % self.args.change_task_interval
            else:
                t_eps = self.t_env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=t_eps, task=self.task, test_mode=test_mode, pred_task=pred_task, mode=mode)
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.reset_epsilon:
            t_eps = self.t_env % self.args.change_task_interval
        else:
            t_eps = self.t_env
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=t_eps, task=self.task, test_mode=test_mode, pred_task=pred_task, mode=mode)

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        if test_mode:
            if mode:
                log_prefix = f"test_{self.args.train_tasks.index(self.task)}_{mode}_"
            else:
                log_prefix = f"test_{self.args.train_tasks.index(self.task)}_"
        else:
            log_prefix = f"{self.args.train_tasks.index(self.task)}_"
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            cur_returns_ = deepcopy(cur_returns)
            cur_stats_ = deepcopy(cur_stats)
            self._log(cur_returns, cur_stats, log_prefix)
            if "battle_won" in cur_stats_:
                return cur_returns_, [cur_stats_["battle_won"] / cur_stats_["n_episodes"]]
            else:
                return cur_returns_, None
        elif (not test_mode) and (self.t_env - self.log_train_stats_t >= self.args.runner_log_interval):
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def probing(self, task2center_z):
        with th.no_grad():
            if len(task2center_z) > 1:
                self.reset()
                terminated = False
                self.mac.init_hidden(batch_size=self.batch_size, task=self.task)
                while not terminated:
                    pre_transition_data = {
                        "state": [self.env.get_state()],
                        "avail_actions": [self.env.get_avail_actions()],
                        "obs": [self.env.get_obs()]
                    }
                    self.batch.update(pre_transition_data, ts=self.t)
                    actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=False, probe_mode=True)
                    reward, terminated, env_info = self.env.step(actions[0])
                    post_transition_data = {
                        "actions": actions,
                        "reward": [(reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.batch.update(post_transition_data, ts=self.t)
                    self.t += 1
                last_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()]
                }
                self.batch.update(last_data, ts=self.t)
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=False, probe_mode=True)
                self.batch.update({"actions": actions}, ts=self.t)
                
                state = self.batch["state"][:,:-1,:]
                if hasattr(self.mac, 'state2feature_transformer'):
                    state = self.mac.state2feature_transformer(state, self.mac.task2decomposer[self.task], self.mac.task2n_agents[self.task])
                terminated = self.batch["terminated"][:, :-1].float()
                mask = self.batch["filled"][:, :-1].float()
                mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
                
                _, mu_lgvar = self.mac.global_encoder(state, mask)
                mu = mu_lgvar[:, : self.args.z_dim]

                obs = self.batch["obs"][:,:-1,:,:]
                bs, seq_len = obs.shape[0], obs.shape[1]
                n_agents = obs.shape[2]
                obs = obs.reshape(bs, seq_len, n_agents, -1).permute(0,2,1,3).reshape(bs * n_agents, seq_len, -1)
                mask = mask.repeat(n_agents, 1, 1)

                if hasattr(self.mac, 'obs2feature'):
                    obs = self.mac.obs2feature(obs, self.task).reshape(bs * n_agents, seq_len, -1)
                _, all_agents_mu_lgvar = self.mac.individual_encoder(obs, mask)
                all_agents_mu = all_agents_mu_lgvar[:, : self.args.z_dim]
                
                task_mu_lst = list(task2center_z.items())

                dist_lst = [((mu - task_mu[1]) ** 2).sum() for task_mu in task_mu_lst]
                closet_dist = min(dist_lst)
                closet_task_idx = dist_lst.index(closet_dist)
                pred_task = task_mu_lst[closet_task_idx][0]
                pred_task_lst = [ (pred_task, closet_dist, mu[0]) ]

                for agent_id in range(n_agents):
                    mu = all_agents_mu[agent_id]
                    
                    dist_lst = [((mu - task_mu[1]) ** 2).sum() for task_mu in task_mu_lst]

                    closet_dist = min(dist_lst)
                    closet_task_idx = dist_lst.index(closet_dist)
                    pred_task = task_mu_lst[closet_task_idx][0]
                    pred_task_lst.append((pred_task, closet_dist, mu))
                return pred_task_lst
            return None

    def get_mu(self, batch):
        with th.no_grad():
            state = batch["state"][:,:-1,:]
            if hasattr(self.mac, 'state2feature_transformer'):
                state = self.mac.state2feature_transformer(state, self.mac.task2decomposer[self.task], self.mac.task2n_agents[self.task])
            terminated = self.batch["terminated"][:, :-1].float()
            mask = self.batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            
            z, mu_lgvar = self.mac.global_encoder(state, mask)
            mu    = mu_lgvar[:, : self.args.z_dim]
            
            obs = self.batch["obs"][:,:-1,:,:]
            bs, seq_len = obs.shape[0], obs.shape[1]
            n_agents = obs.shape[2]
            obs = obs.reshape(bs, seq_len, n_agents, -1).permute(0,2,1,3).reshape(bs * n_agents, seq_len, -1)
            mask = mask.repeat(n_agents, 1, 1)

            if hasattr(self.mac, 'obs2feature'):
                obs = self.mac.obs2feature(obs, self.task).reshape(bs * n_agents, seq_len, -1)
            individual_z, all_agents_mu_lgvar = self.mac.individual_encoder(obs, mask)
                
            return mu, z, individual_z

    def run_identify(self, test_mode=False, pred_task=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=test_mode, pred_task=pred_task, mode='ctce')
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=test_mode, pred_task=pred_task, mode='ctce')
        self.batch.update({"actions": actions}, ts=self.t)

        return self.batch
