import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
class BaselineNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BaselineNN, self).__init__()

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.input_dim = input_dim
        self.baseline_net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.baseline_net.apply(init_weights)
        
    def forward(self, x):
        assert(x.shape[1]) == self.input_dim
        out = self.baseline_net(x.float())
        return out
