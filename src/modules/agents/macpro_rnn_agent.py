import torch as th
import copy
import random
from .rnn_agent import RNNAgent

class MACProRNNAgent(RNNAgent):
    def __init__(self, task2input_shape_info, tasks2args, main_args):
        
        super(MACProRNNAgent, self).__init__(task2input_shape_info, tasks2args, main_args)

    def forward(self, inputs, hidden_state, task, test_mode, probe_mode, pred_task=None, mode=None):
        
        feature = self.feature_layer(inputs, hidden_state)

        if probe_mode and len(self.task2head) >= 2:
            with th.no_grad():
                random_head = random.choice(list(self.task2head.values()))
                q = random_head(feature)
                return q, feature

        if test_mode:
            with th.no_grad():
                if mode == 'random_head' and len(self.task2head) > 0:
                    random_head = random.choice(list(self.task2head.values()))
                    q = random_head(feature)
                    return q, feature
                elif mode == 'oracle' and task in self.task2head:
                    q = self.task2head[task](feature)
                    return q, feature
                elif self.main_args.probe_nepisode > 0:
                    if pred_task and mode == 'ctde':
                        most_pred_task = max(set(pred_task), key=pred_task.count)
                        q = self.task2head[most_pred_task](feature)
                        if len(set(pred_task)) > 1:
                            n_agents = feature.shape[0]
                            for agent_id in range(n_agents):
                                pred_task_this_agent = pred_task[agent_id]
                                q_this_agent = self.task2head[pred_task_this_agent](feature)
                                q[agent_id] = q_this_agent[agent_id]
                    else:
                        q = self.head(feature)
                    return q, feature

        q = self.head(feature)
        
        return q, feature

    def save_head(self, task):
        self.task2head[task] = copy.deepcopy(self.head)

    def reset_head(self):
        reset_recursive(self.head)
    
    def load_head(self, task):
        assert task in self.task2head
        self.head.load_state_dict(self.task2head[task].state_dict())

def reset_recursive(x):
    if len(list(x.children())) == 0:
        if hasattr(x, 'reset_parameters'):
            x.reset_parameters()
        return
    for layer in x.children():
        reset_recursive(layer)
