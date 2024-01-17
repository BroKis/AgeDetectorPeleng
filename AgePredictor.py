import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
batch_size = 64
class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        #self.conv2 = nn.Conv2d(32, batch_size, 3, padding=1)
        self.fc1 = nn.Linear(batch_size * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
       # x = F.relu(self.conv2(x))
        x = x.view(-1, batch_size * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class AgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return (torch.abs(outputs - targets)).mean()