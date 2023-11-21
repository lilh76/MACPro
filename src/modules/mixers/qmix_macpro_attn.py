import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerMACPro(nn.Module):
    def __init__(self, surrogate_decomposer, main_args):
        super(QMixerMACPro, self).__init__()
        self.main_args = main_args
        self.embed_dim = main_args.mixing_embed_dim
        self.attn_embed_dim = main_args.attn_embed_dim
        self.entity_embed_dim = main_args.entity_embed_dim

        self.surrogate_decomposer = surrogate_decomposer
        state_nf_al, state_nf_en, timestep_state_dim = \
            surrogate_decomposer.state_nf_al, surrogate_decomposer.state_nf_en, surrogate_decomposer.timestep_number_state_dim
        self.state_last_action, self.state_timestep_number = surrogate_decomposer.state_last_action, surrogate_decomposer.state_timestep_number

        self.n_actions_no_attack = surrogate_decomposer.n_actions_no_attack

        if self.state_last_action:
            self.ally_encoder = nn.Linear(state_nf_al + self.n_actions_no_attack + 1, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en, self.entity_embed_dim)
        else:
            self.ally_encoder = nn.Linear(state_nf_al, self.entity_embed_dim)
            self.enemy_encoder = nn.Linear(state_nf_en, self.entity_embed_dim)
        
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        
        mixing_input_dim = self.entity_embed_dim + timestep_state_dim

        self.hyper_w_1 = nn.Linear(mixing_input_dim, self.embed_dim)
        self.hyper_w_final = nn.Linear(mixing_input_dim, self.embed_dim)
        
        self.hyper_b_1 = nn.Linear(mixing_input_dim, self.embed_dim)
        
        self.V = nn.Sequential(nn.Linear(mixing_input_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))


    def forward(self, agent_qs, states, task_decomposer):
        bs, seq_len, n_agents = agent_qs.size(0), agent_qs.size(1), agent_qs.size(2)
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        ally_states, enemy_states, last_action_states, timestep_number_state = task_decomposer.decompose_state(states)
        ally_states = th.stack(ally_states, dim=0)
        enemy_states = th.stack(enemy_states, dim=0)

        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(last_action_states)
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        ally_embed = self.ally_encoder(ally_states)
        enemy_embed = self.enemy_encoder(enemy_states)

        entity_embed = th.cat([ally_embed, enemy_embed], dim=0)

        proj_query = self.query(entity_embed).permute(1, 2, 0, 3).reshape(bs*seq_len, n_entities, self.attn_embed_dim)
        proj_key = self.key(entity_embed).permute(1, 2, 3, 0).reshape(bs*seq_len, self.attn_embed_dim, n_entities)
        energy = th.bmm(proj_query, proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(bs*seq_len, self.entity_embed_dim, n_entities)
        attn_out = th.bmm(proj_value, attn_score).mean(dim=-1).reshape(bs, seq_len, self.entity_embed_dim)

        if self.state_timestep_number:
            raise Exception(f"Not Implemented")
        else:
            mixing_input = attn_out[:, :, None, :].repeat(1, 1, n_agents, 1)
        
        w1 = th.abs(self.hyper_w_1(mixing_input))
        b1 = self.hyper_b_1(mixing_input).mean(dim=2)
        w1 = w1.view(-1, n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        agent_qs = agent_qs.view(-1, 1, n_agents)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        w_final = th.abs(self.hyper_w_final(mixing_input)).mean(dim=2).view(-1, self.embed_dim, 1)
        v = self.V(mixing_input).mean(dim=2).view(-1, 1, 1)

        y = th.bmm(hidden, w_final) + v

        q_tot = y.view(bs, -1, 1)
        
        return q_tot

