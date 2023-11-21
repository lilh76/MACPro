import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args, input_shape, output_shape):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, output_shape)
        )

    def forward(self, inputs):

        return self.fc(inputs)
