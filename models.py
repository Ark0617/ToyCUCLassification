import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class ToyConfPredictor(nn.Module):
    def __init__(self, teacher_hidden_dim, hidden_dim):
        super(ToyConfPredictor, self).__init__()
        self.fc1 = nn.Linear(3 + teacher_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        # self.d1 = nn.Dropout(0.5)
        # self.d2 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ToyTeacherNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ToyTeacherNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

    def feature_extract(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ToyStudentNet(nn.Module):
    def __init__(self, hidden_dim):
        super(ToyStudentNet, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)

    def feature_extract(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x