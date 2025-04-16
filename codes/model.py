import torch.nn as nn



class SimpleTanhNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleTanhNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)