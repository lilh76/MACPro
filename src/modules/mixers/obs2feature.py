import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.embed import binary_embed
from collections import OrderedDict

class ObsAction2Feature(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(ObsAction2Feature, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        obs_en_dim += 1
        self.query       = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim)
        self.ally_key    = nn.Linear(obs_al_dim,          self.attn_embed_dim)
        self.ally_value  = nn.Linear(obs_al_dim,          self.entity_embed_dim)
        self.enemy_key   = nn.Linear(obs_en_dim,          self.attn_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim,          self.entity_embed_dim)
        self.own_value   = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.rnn = nn.Sequential(
            nn.Linear(self.entity_embed_dim * 3, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.entity_embed_dim)
        )

    def forward(self, inputs, task):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        action_shape = self.task2last_action_shape[task]
        obs_dim = task_decomposer.obs_dim
        bs_n_agents, seq_len = inputs.shape[0], inputs.shape[1]
        obs_inputs    = inputs[:, :, : obs_dim].reshape(bs_n_agents * seq_len, -1)
        action_inputs = inputs[:, :,   obs_dim : obs_dim + action_shape].reshape(bs_n_agents * seq_len, -1)
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)
        bs = int(own_obs.shape[0] / task_n_agents)
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(action_inputs)
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)
        own_hidden = self.own_value(own_obs)
        query = self.query(own_obs)
        ally_keys = self.ally_key(ally_feats).permute(1, 2, 0)
        enemy_keys = self.enemy_key(enemy_feats).permute(1, 2, 0)
        ally_values = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_values = self.enemy_value(enemy_feats).permute(1, 0, 2)
        ally_hidden = attention(query, ally_keys, ally_values, self.attn_embed_dim)
        enemy_hidden = attention(query, enemy_keys, enemy_values, self.attn_embed_dim)        
        tot_hidden = th.cat([own_hidden, ally_hidden, enemy_hidden], dim=-1)
        e = self.rnn(tot_hidden)
        return e

class Obs2Feature(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(Obs2Feature, self).__init__()
        self.task2last_action_shape = {task: task2input_shape_info[task]["last_action_shape"] for task in task2input_shape_info}
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        wrapped_obs_own_dim = obs_own_dim + args.id_length
        self.query       = nn.Linear(wrapped_obs_own_dim, self.attn_embed_dim)
        self.ally_key    = nn.Linear(obs_al_dim,          self.attn_embed_dim)
        self.ally_value  = nn.Linear(obs_al_dim,          self.entity_embed_dim)
        self.enemy_key   = nn.Linear(obs_en_dim,          self.attn_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim,          self.entity_embed_dim)
        self.own_value   = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.rnn = nn.Sequential(
            nn.Linear(self.entity_embed_dim * 3, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.entity_embed_dim)
        )

    def forward(self, inputs, task):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        bs_n_agents, seq_len, obs_shape = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        obs_inputs = inputs.reshape(bs_n_agents * seq_len, -1)
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)
        bs = int(own_obs.shape[0] / task_n_agents)
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        own_obs = th.cat([own_obs, agent_id_inputs], dim=-1)
        enemy_feats = th.stack(enemy_feats, dim=0)
        ally_feats = th.stack(ally_feats, dim=0)
        own_hidden = self.own_value(own_obs)
        query = self.query(own_obs)
        ally_keys = self.ally_key(ally_feats).permute(1, 2, 0)
        enemy_keys = self.enemy_key(enemy_feats).permute(1, 2, 0)
        ally_values = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_values = self.enemy_value(enemy_feats).permute(1, 0, 2)
        ally_hidden = attention(query, ally_keys, ally_values, self.attn_embed_dim)
        enemy_hidden = attention(query, enemy_keys, enemy_values, self.attn_embed_dim)        
        tot_hidden = th.cat([own_hidden, ally_hidden, enemy_hidden], dim=-1)
        e = self.rnn(tot_hidden)
        state_shape_info = th.ones_like(e[:,0]).unsqueeze(-1) * obs_shape
        res = th.cat((e, state_shape_info), dim = -1)
        return res

def attention(q, k, v, attn_dim):
    """
        q: [bs*n_agents, attn_dim]
        k: [bs*n_agents, attn_dim, n_entity]
        v: [bs*n_agents, n_entity, value_dim]
    """
    energy = th.bmm(q.unsqueeze(1)/(attn_dim ** (1/2)), k)
    attn_score = F.softmax(energy, dim=-1) 
    attn_out = th.bmm(attn_score, v).squeeze(1)
    return attn_out
