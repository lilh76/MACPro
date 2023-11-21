import torch as th
import torch.nn as nn
import torch.nn.functional as F

class State2Feature(nn.Module):
    def __init__(self, surrogate_decomposer, main_args):
        super(State2Feature, self).__init__()
        self.main_args = main_args
        self.embed_dim = main_args.mixing_embed_dim
        self.attn_embed_dim = main_args.attn_embed_dim
        self.entity_embed_dim = main_args.entity_embed_dim

        self.surrogate_decomposer = surrogate_decomposer
        state_nf_al = surrogate_decomposer.state_nf_al
        state_nf_en = surrogate_decomposer.state_nf_en
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

    def forward(self, states, task_decomposer, n_agents):
        bs, seq_len, state_shape = states.shape[0], states.shape[1], states.shape[2]
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
        state_shape_info = th.ones_like(attn_out[:,:,0]).unsqueeze(-1) * state_shape
        res = th.cat((attn_out, state_shape_info), dim = -1)
        return res
