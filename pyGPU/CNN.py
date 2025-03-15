import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        # 第一畳み込み層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 第二畳み込み層
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # プーリング層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全結合層
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 一次元に変換
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # 最終出力
        x = self.fc2(x)
        return x

model = SimpleCNN(num_classes=10)
print(model)