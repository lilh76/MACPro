import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, task2input_shape_info, tasks2args, main_args):
        
        super(RNNAgent, self).__init__()
        self.main_args = main_args
        self.tasks2args = tasks2args

        task = main_args.train_tasks[0]
        input_shape = task2input_shape_info[task]["input_shape"]
        
        self.feature_layer = FeatureLayer(input_shape, main_args)

        self.head = Head(tasks2args[task].n_actions, main_args)

        self.task2head = {}

    def init_hidden(self):
        return self.feature_layer.fc1.weight.new(1, self.main_args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, task, test_mode, probe_mode, pred_task=None, mode=None):

        feature = self.feature_layer(inputs, hidden_state) # [bs * n_agents, h_dim]

        q = self.head(feature)
        
        return q, feature

    def save_head(self, task):
        pass

class FeatureLayer(nn.Module):
    def __init__(self, input_shape, args):
        super(FeatureLayer, self).__init__()
        self.hidden_dim = args.rnn_hidden_dim
        self.fc1 = nn.Linear(input_shape, self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc3 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc4 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.fc5 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        return h

class Head(nn.Module):
    def __init__(self, n_actions, args):
        super(Head, self).__init__()
        self.hidden_dim = args.rnn_hidden_dim
        self.fc5 = nn.Linear(self.hidden_dim, n_actions)

    def forward(self, feature):
        q = self.fc5(feature)
        return q
        