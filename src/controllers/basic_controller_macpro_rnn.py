from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from modules.encoders.mlp import MLP
from modules.encoders.transformer import TransformerEncoder
import os
from copy import deepcopy

# This multi-agent controller shares parameters between agents
class BasicMACMACProRNN:
    def __init__(self, train_tasks, task2scheme, task2args, main_args):
        # set some task-specific attributes
        self.train_tasks = train_tasks
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.task2n_agents = {task: self.task2args[task].n_agents for task in train_tasks}
        self.main_args = main_args

        # set some common attributes
        self.agent_output_type = main_args.agent_output_type
        self.action_selector = action_REGISTRY[main_args.action_selector](main_args)
        
        self.task2input_shape_info = self._get_input_shape()
        self._build_agents(self.task2input_shape_info)

        self.task = train_tasks[0]

        self.hidden_states = None

        self.task2center_z = {}
        self.task2radius = {}

        state_shape = task2scheme[self.task]['state']['vshape']
        obs_shape = task2scheme[self.task]['obs']['vshape']
        n_actions = task2scheme[self.task]['avail_actions']['vshape'][0]
        n_agents = self.task2n_agents[self.task]

        self.global_encoder     = TransformerEncoder(main_args, state_shape)
        self.individual_encoder = TransformerEncoder(main_args, obs_shape)

        self.world_model = MLP(main_args, state_shape + (n_actions + obs_shape) * n_agents + main_args.z_dim, 
                                          state_shape +              obs_shape  * n_agents + 1)

    def select_actions(self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False, probe_mode=False, pred_task=None, mode=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, task, t_ep, test_mode=test_mode, probe_mode=probe_mode, pred_task=pred_task, mode=mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, task, t, test_mode=False, probe_mode=False, pred_task=None, mode=None):

        n_agents = self.task2n_agents[task]

        agent_inputs = self._build_inputs(ep_batch, t, task)

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, task, test_mode, probe_mode, pred_task, mode=mode)

        return agent_outs.view(ep_batch.batch_size, n_agents, -1)

    def init_hidden(self, batch_size, task):
        n_agents = self.task2n_agents[task]
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.world_model.cuda()
        self.global_encoder.cuda()
        self.individual_encoder.cuda()

    def save_models(self, path):

        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if hasattr(self.agent, 'task2head'):
            for task, head in self.agent.task2head.items():
                task_str = task.replace(":", '!')
                th.save(head.state_dict(), f"{path}/head___{task_str}.th")
        th.save(self.global_encoder.state_dict(), "{}/global_encoder.th".format(path))
        th.save(self.individual_encoder.state_dict(), "{}/individual_encoder.th".format(path))
        th.save(self.world_model.state_dict(), "{}/world_model.th".format(path))

    def load_models(self, path):

        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        for model in os.listdir(path):
            if "head" in model:
                task_str = model.split("___")[1][:-3]
                task_str = task_str.replace("!", ':')
                self.agent.task2head[task_str] = deepcopy(self.agent.head)
                self.agent.task2head[task_str].load_state_dict(th.load(f"{path}/{model}", map_location=lambda storage, loc: storage))
        self.global_encoder.load_state_dict(th.load("{}/global_encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.individual_encoder.load_state_dict(th.load("{}/individual_encoder.th".format(path), map_location=lambda storage, loc: storage))
        self.world_model.load_state_dict(th.load("{}/world_model.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.main_args.agent](task2input_shape_info, self.task2args, self.main_args)
        
    def _build_inputs(self, batch, t, task):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        # get args, n_agents for this specific task
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        if task_args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if task_args.obs_agent_id:
            inputs.append(th.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        
        inputs = th.cat([x.reshape(bs*n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.train_tasks:
            task_scheme = self.task2scheme[task]
            input_shape = task_scheme["obs"]["vshape"]
            last_action_shape, agent_id_shape = 0, 0
            if self.task2args[task].obs_last_action:
                input_shape += task_scheme["actions_onehot"]["vshape"][0]
                last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            if self.task2args[task].obs_agent_id:
                input_shape += self.task2n_agents[task]
                agent_id_shape = self.task2n_agents[task]
            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
            }
        return task2input_shape_info
