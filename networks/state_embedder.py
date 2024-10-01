import torch

class StateEmbedder(torch.nn.Module):
    def __init__(self, state_dim: int, hdim: int):
        super(StateEmbedder, self).__init__()
        self.state_dim = state_dim
        self.hdim = hdim

        self.fc1 = torch.nn.Linear(self.state_dim, self.hdim)
        self.relu = torch.nn.ReLU()

    def forward(self, state):
        return self.relu(self.fc1(state))
