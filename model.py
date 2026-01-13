import torch
from torch import nn
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #OH = ((H+2p-FH)/S)+1
        # out_cahnnels:卷积核的数量  in_cahnnels=1:代表输入的是灰度图，kernel_size=5:代表卷积核的大小是5x5, padding=2:填充为2
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)#默认步幅为1，不用声明
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    # print(summary(model, (1, 28, 28)))
