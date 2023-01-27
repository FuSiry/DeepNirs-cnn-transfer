import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable




class conv1(nn.Module):
    def __init__(self):
        super(conv1,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=17, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=15, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.AvgMaxPool = nn.AdaptiveMaxPool1d(70)
        self.fc = nn. Linear(4480,1) #8960 ,17920
        self.drop = nn.Dropout(0.2)

    def forward(self,out):
      out = self.conv1(out)
      # out = self.drop(out)
      out = self.conv2(out)
      out = self.conv3(out)
      out = F.relu(self.AvgMaxPool(out))
      out = out.view(out.size(0),-1)
      out = self.fc(out)
      return out