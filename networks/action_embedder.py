import torch

class ActionEmbedder(torch.nn.Module):
    def __init__(self, act_dim: int, hdim: int):
        super(ActionEmbedder, self).__init__()
        self.act_dim = act_dim
        self.hdim = hdim

        self.fc1 = torch.nn.Linear(self.act_dim, self.hdim)
        self.relu = torch.nn.ReLU()

    def forward(self, state):
        return self.relu(self.fc1(state))
