import torch as th
import torch.nn as nn
import torch.nn.functional as F
from utils.embed import binary_embed
from collections import OrderedDict

class AttnAgent(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, main_args):
        
        super(AttnAgent, self).__init__()
        self.main_args = main_args
        self.feature_layer = FeatureLayer(task2input_shape_info, task2decomposer, task2n_agents, decomposer, main_args)
        self.head = Head(task2input_shape_info, task2decomposer, task2n_agents, decomposer, main_args)
        self.task2head = {}

    def init_hidden(self):
        return self.head.wo_action_layer.fc1.weight.new(1, self.main_args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, task, test_mode, probe_mode, pred_task=None, mode=None):

        feature = self.feature_layer(inputs, hidden_state, task)

        q = self.head(feature)
        
        return q, feature[0]

    def save_head(self, task):
        pass

class FeatureLayer(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(FeatureLayer, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(self.entity_embed_dim * 3, self.entity_embed_dim * 6),
            nn.ReLU(),
            nn.Linear(self.entity_embed_dim * 6, self.entity_embed_dim * 6),
            nn.ReLU(),
            nn.Linear(self.entity_embed_dim * 6, self.entity_embed_dim * 3)
        )
        self.rnn = nn.GRUCell(self.entity_embed_dim * 3, args.rnn_hidden_dim)
        self.enemy_embed = nn.Sequential(
            nn.Linear(obs_en_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )

    def forward(self, inputs, hidden_state, task):
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, _ = inputs[:, :obs_dim], \
            inputs[:, obs_dim:obs_dim+last_action_shape], inputs[:, obs_dim+last_action_shape:]
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(obs_inputs)    # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0]/task_n_agents)
        agent_id_inputs = [th.as_tensor(binary_embed(i + 1, self.args.id_length, self.args.max_agent), dtype=own_obs.dtype) for i in range(task_n_agents)]
        agent_id_inputs = th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        _, attack_action_info, compact_action_states = task_decomposer.decompose_action_info(last_action_inputs)
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1) # [bs * n_agents, 16]
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1) # [bs * n_enemy, n_agents, 6]
        ally_feats = th.stack(ally_feats, dim=0) # [bs * n_ally, n_agents, 5]
        own_hidden = self.own_value(own_obs)
        query = self.query(own_obs)
        ally_keys = self.ally_key(ally_feats).permute(1, 2, 0)
        enemy_keys = self.enemy_key(enemy_feats).permute(1, 2, 0)
        ally_values = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_values = self.enemy_value(enemy_feats).permute(1, 0, 2)
        ally_hidden = self.attention(query, ally_keys, ally_values, self.attn_embed_dim)
        enemy_hidden = self.attention(query, enemy_keys, enemy_values, self.attn_embed_dim)        
        tot_hidden = th.cat([own_hidden, ally_hidden, enemy_hidden], dim=-1)
        tot_hidden = self.fc(tot_hidden)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(tot_hidden, h_in)
        enemy_embed = self.enemy_embed(enemy_feats)

        return [h, enemy_embed, enemy_feats]

    def attention(self, q, k, v, attn_dim):
        """
            q: [bs*n_agents, attn_dim]
            k: [bs*n_agents, attn_dim, n_entity]
            v: [bs*n_agents, n_entity, value_dim]
        """
        energy = th.bmm(q.unsqueeze(1)/(attn_dim ** (1/2)), k)
        attn_score = F.softmax(energy, dim=-1) 
        attn_out = th.bmm(attn_score, v).squeeze(1)
        return attn_out

class Head(nn.Module):
    def __init__(self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args):
        super(Head, self).__init__()
        n_actions_no_attack = decomposer.n_actions_no_attack
        self.wo_action_layer = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(args.rnn_hidden_dim, n_actions_no_attack))
        ]))
        self.attack_action_layer = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(args.rnn_hidden_dim * 2, args.rnn_hidden_dim)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)),
            ("relu2", nn.ReLU()),
            ("fc3", nn.Linear(args.rnn_hidden_dim, 1))
        ]))

    def forward(self, feature):
        h, enemy_hidden, enemy_feats = feature[0], feature[1], feature[2]
        # compute wo_action_q
        wo_action_q = self.wo_action_layer(h)
        attack_action_input = th.cat([enemy_hidden, h.unsqueeze(0).repeat(enemy_feats.size(0), 1, 1)], dim=-1)
        attack_action_q = self.attack_action_layer(attack_action_input).squeeze(-1).transpose(0, 1)
        
        q = th.cat([wo_action_q, attack_action_q], dim=-1)

        return q
        
