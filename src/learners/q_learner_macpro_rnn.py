import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix_macpro_fc import QMixerMACProFC as QMixer
import torch as th
from torch.optim import RMSprop, Adam
import random
import torch.distributions as D
from torch.distributions import kl_divergence

class QLearnerMACProRNN:
    def __init__(self, mac, logger, main_args):
        self.args = main_args
        self.mac = mac
        self.logger = logger

        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "qmix":
                self.mixer = QMixer(main_args, self.task2args)
            else:
                raise ValueError("Mixer {} not recognised.".format(main_args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=main_args.lr, alpha=main_args.optim_alpha, eps=main_args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.prev_feature_layer = {}

        self.global_params     = list(self.mac.global_encoder.parameters()) + \
                                 list(self.mac.world_model.parameters())
        self.individual_params = self.mac.individual_encoder.parameters()

        self.global_optimiser     = Adam(params=self.global_params,     lr=main_args.lr)
        self.individual_optimiser = Adam(params=self.individual_params, lr=main_args.lr)

        self.z_dim = main_args.z_dim

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task=None, is_first_task=False):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        reg_loss = 0
        if self.args.reg and not is_first_task:
            reg_losses = self.reg_penalty()
            reg_loss = sum([loss for task, loss in reg_losses.items()])
            self.logger.log_stat("forget_loss", reg_loss.item(), t_env)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, task, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, task, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        td_loss = loss.detach().clone()

        if self.args.reg and not is_first_task:
            loss += reg_loss * self.args.reg_coef

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            n_agents = self.task2n_agents[task]
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".  format(path), map_location=lambda storage, loc: storage))

    def reg_init(self, task):
        self.prev_feature_layer = {}
        self.prev_feature_layer[task] = {n: p.detach().clone() for n, p in self.mac.agent.feature_layer.named_parameters()}

    def reg_penalty(self):
        loss = {}
        for task in self.prev_feature_layer.keys():
            loss[task] = 0
            task_param = self.prev_feature_layer[task]
            for n, p in self.mac.agent.feature_layer.named_parameters():
                loss[task] += ((p - task_param[n]) ** 2).sum()
        return loss

    def train_probing(self, probing_batch_dict, t_env):

        rec_loss         = th.zeros(1).to(self.args.device)
        approx_loss      = th.zeros(1).to(self.args.device)

        global_z_dict = {}
        individual_z_dict = {}
        global_mu_lgvar_dict = {}
        individual_mu_lgvar_dict = {}
        
        bs = 0
        for task, batch_lst in probing_batch_dict.items():

            state = th.cat([batch["state"] for batch in batch_lst])
            bs, seq_len = state.shape[0], state.shape[1] - 1
            obs = th.cat([batch["obs"] for batch in batch_lst]).reshape(bs, seq_len + 1, -1)
            reward = th.cat([batch["reward"] for batch in batch_lst])[:,:-1,:]
            terminated = th.cat([batch["terminated"] for batch in batch_lst])[:, :-1].float()
            mask = th.cat([batch["filled"] for batch in batch_lst])[:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

            traj_z, mu_lgvar = self.mac.global_encoder(state[:,:-1,:], mask)
            mu = mu_lgvar[:, : self.z_dim]

            center = mu.mean(dim = 0)
            self.mac.task2center_z[task] = center
            center = center.unsqueeze(0).repeat(bs, 1)
            radius = th.sqrt(th.sum((mu - center) ** 2, dim = -1)).mean()
            self.mac.task2radius[task] = radius
            global_z_dict[task] = traj_z
            global_mu_lgvar_dict[task] = mu_lgvar

            if self.args.rec_coef > 0:
                model_target = th.cat((state[:,1: ,:] - state[:,:-1,:], obs[:,1: ,:], reward), dim = -1)
                action = th.cat([batch["actions_onehot"] for batch in batch_lst])[:,:-1,:].reshape(bs, seq_len, -1)
                z4model = th.cat([traj_z.unsqueeze(1) for _ in range(seq_len)], dim = 1)
                model_inputs = th.cat((state[:,:-1,:], obs[:,:-1,:], action, z4model), dim = -1)
                model_rec = self.mac.world_model(model_inputs)
                rec_loss_ = ((model_rec - model_target.detach().clone()) ** 2).mean(dim = -1).unsqueeze(-1)
                masked_rec_loss = rec_loss_ * mask
                rec_loss += masked_rec_loss.sum() / mask.sum()

            n_agents = self.task2n_agents[task]
            obs = obs[:,:-1,:]
            obs = obs.reshape(bs, seq_len, n_agents, -1).permute(0,2,1,3).reshape(bs * n_agents, seq_len, -1)
            mask = mask.unsqueeze(1).repeat(1, n_agents, 1, 1).reshape(bs * n_agents, seq_len, -1)
            individual_z, individual_mu_lgvar = self.mac.individual_encoder(obs, mask)
            individual_mu_lgvar_dict[task] = individual_mu_lgvar

            global_mu_lgvar = mu_lgvar.unsqueeze(1).repeat(1, n_agents, 1).reshape(bs * n_agents, -1)

            approx_loss += self.calculate_kl(global_mu_lgvar.detach().clone(), individual_mu_lgvar)

            individual_z_dict[task] = individual_z.reshape(bs, n_agents, self.args.z_dim).permute(1,0,2)

        rec_loss         /= len(probing_batch_dict)
        approx_loss      /= len(probing_batch_dict)
        self.logger.log_stat(        "rec_loss",         rec_loss.item(), t_env)
        self.logger.log_stat(     "approx_loss",      approx_loss.item(), t_env)

        pos_loss = th.zeros(1).to(self.args.device)
        neg_loss = th.zeros(1).to(self.args.device)
        if self.args.contrastive_coef_global > 0 and len(probing_batch_dict) >= 2:
            lst = list(global_mu_lgvar_dict.values())
            for i, mu_lgvar in enumerate(lst):
                task_id = (th.ones(bs,1) * i).to(self.args.device)
                mu_lgvar = th.cat((task_id, mu_lgvar), dim = -1)
                lst[i] = mu_lgvar
            mu_lgvar = th.cat(lst)
            random_idx = random.sample([i for i in range(mu_lgvar.shape[0])], bs)
            sampled_mu_lgvar = mu_lgvar[random_idx]
            pos_cnt = 0
            neg_cnt = 0
            for i in range(bs):
                mu_lgvar_i = sampled_mu_lgvar[i]
                task_i     = mu_lgvar_i[0]
                mu_lgvar_i = mu_lgvar_i[1:]
                for j in range(i + 1, bs):
                    mu_lgvar_j = sampled_mu_lgvar[j]
                    task_j     = mu_lgvar_j[0]
                    mu_lgvar_j = mu_lgvar_j[1:]
                    if task_i == task_j:
                        pos_loss +=      self.calculate_kl(mu_lgvar_i.unsqueeze(0), mu_lgvar_j.unsqueeze(0))
                        pos_cnt  += 1
                    else:
                        neg_loss += 1 / (self.calculate_kl(mu_lgvar_i.unsqueeze(0), mu_lgvar_j.unsqueeze(0)) + 1e-3)
                        neg_cnt  += 1
            pos_loss /= (pos_cnt + 1e-3)
            neg_loss /= (neg_cnt + 1e-3)
            self.logger.log_stat("pos_loss", pos_loss.item(), t_env)
            self.logger.log_stat("neg_loss", neg_loss.item(), t_env)
        
        oracle_loss = self.args.rec_coef * rec_loss + self.args.contrastive_coef_global * (pos_loss + neg_loss)

        self.global_optimiser.zero_grad()
        oracle_loss.backward()
        th.nn.utils.clip_grad_norm_(self.global_params, self.args.grad_norm_clip)
        self.global_optimiser.step()

        pos_loss_agent = th.zeros(1).to(self.args.device)
        neg_loss_agent = th.zeros(1).to(self.args.device)
        if self.args.contrastive_coef_individual > 0 and len(probing_batch_dict) >= 2:
            lst = list(individual_mu_lgvar_dict.values())
            for i, mu_lgvar in enumerate(lst):
                task_id = (th.ones(bs * n_agents, 1) * i).to(self.args.device)
                mu_lgvar = th.cat((task_id, mu_lgvar), dim = -1)
                lst[i] = mu_lgvar
            mu_lgvar = th.cat(lst)
            random_idx = random.sample([i for i in range(mu_lgvar.shape[0])], bs)
            mu_lgvar = mu_lgvar[random_idx]
            pos_cnt = 0
            neg_cnt = 0
            for i in range(bs):
                mu_lgvar_i = mu_lgvar[i]
                task_i     = mu_lgvar_i[0]
                mu_lgvar_i = mu_lgvar_i[1:]
                for j in range(i + 1, bs):
                    mu_lgvar_j = mu_lgvar[j]
                    task_j     = mu_lgvar_j[0]
                    mu_lgvar_j = mu_lgvar_j[1:]
                    if task_i == task_j:
                        pos_loss_agent +=      self.calculate_kl(mu_lgvar_i.unsqueeze(0), mu_lgvar_j.unsqueeze(0))
                        pos_cnt  += 1
                    else:
                        neg_loss_agent += 1 / (self.calculate_kl(mu_lgvar_i.unsqueeze(0), mu_lgvar_j.unsqueeze(0)) + 1e-3)
                        neg_cnt  += 1
            pos_loss_agent /= (pos_cnt + 1e-3)
            neg_loss_agent /= (neg_cnt + 1e-3)
            self.logger.log_stat("pos_loss_agent", pos_loss_agent.item(), t_env)
            self.logger.log_stat("neg_loss_agent", neg_loss_agent.item(), t_env)

        individual_loss = self.args.approx_coef * approx_loss + self.args.contrastive_coef_individual * (pos_loss_agent + neg_loss_agent)
        self.individual_optimiser.zero_grad()
        individual_loss.backward()
        th.nn.utils.clip_grad_norm_(self.individual_params, self.args.grad_norm_clip)
        self.individual_optimiser.step()

        return

    def calculate_kl(self, mu_lgvar_1, mu_lgvar_2):
        mu_1     = mu_lgvar_1[:, : self.z_dim]
        lgvar_1 = mu_lgvar_1[:,   self.z_dim :]
        std_1    = th.exp(0.5 * lgvar_1)
        mu_2     = mu_lgvar_2[:, : self.z_dim]
        lgvar_2 = mu_lgvar_2[:,   self.z_dim :]
        std_2    = th.exp(0.5 * lgvar_2)
        p_1 = D.Normal(mu_1, std_1)
        p_2 = D.Normal(mu_2, std_2)
        return (kl_divergence(p_1, p_2).mean() + kl_divergence(p_2, p_1).mean()) / 2
