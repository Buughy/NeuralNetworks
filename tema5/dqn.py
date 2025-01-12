import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_dim2=32):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == "__main__":
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)

    result = net(state)
    print(result)

