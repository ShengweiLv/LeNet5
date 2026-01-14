import copy
import time

from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as Data
from torchvision import transforms  #处理数据集
import matplotlib.pyplot as plt
from model import LeNet5
import torch
import pandas as pd

from torch import nn

def train_val_data_process():#处理训练集和验证集
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),#设置输入图片大小
                              download=True)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=2)

    val_loader = Data.DataLoader(dataset=val_data,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=2)
    return train_loader, val_loader

def train_model_process(model, train_loader, val_loadern, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #优化器，梯度下降法更新参数

    criterion = nn.CrossEntropyLoss() #分类中一般用交叉熵损失函数，回归中一般用均方误差
    #将模型放入训练设备中
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    #最高准确度
    best_acc = 0.0
    #训练集损失函数列表
    train_loss_all = []
    # 验证集损失函数列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    #当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))#0~99,但传入的是100
        print('-' * 10)

        #初始化参数
        #训练集损失函数
        train_loss = 0.0
        #训练集准确度
        train_acc = 0.0
        #验证集损失函数
        val_loss = 0.0
        #验证集准确度
        val_acc = 0.0

        #训练集样本数量
        train_num = 0
        #验证集样本数量
        val_num = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            #将特征和标签放到设备中
            data, target = data.to(device), target.to(device)
            #设置模型为训练模式
            model.train()

            #前向传播过程，输入为一个batch，输出为一个batch的预测
            output = model(data)
            #查找每一行中最大值对应的行标  因为最后会输出10个值
            pre_label = torch.argmax(output, dim=1)

            #计算每一个batch的损失函数
            loss = criterion(output, target)#取出output中下标为target的值，为分子上的指数上的x，所有的加起来为分母，这是交叉熵函数内置的softmax归一化，然后再取负的对数，就得到loss值

            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播计算
            loss.backward()
            #根据反向传播的梯度信息来更新网络的参数，已起到降低loss函数计算值的作用
            optimizer.step()

            #对损失函数进行累加
            train_loss += loss.item()*data.size(0)
            #如果预测正确，则准确度train_acc＋1
            train_acc += torch.sum(pre_label == target)
            train_num += data.size(0)

        for batch_idx, (data, target) in enumerate(val_loadern):
            data, target = data.to(device), target.to(device)
            #设置模型为评估模式
            model.eval()
            #前向传播过程，输入一个batch，输出一个batch对应的预测
            output = model(data)
            #找最大可能
            pre_label = torch.argmax(output, dim=1)

            # 计算每一个batch的损失函数
            loss = criterion(output, target)

            # 对损失函数进行累加
            val_loss += loss.item() * data.size(0)
            # 如果预测正确，则准确度train_acc＋1
            val_acc += torch.sum(pre_label == target)
            val_num += data.size(0)

        #计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)

        # 计算并保存训练集的准确率
        train_acc_all.append(train_acc.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)

        # 计算并保存验证集的准确率
        val_acc_all.append(val_acc.double().item() / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch,train_loss_all[-1],train_acc_all[-1]))#-1是指最后一个值
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        #寻找最高准确度的权重
        if val_acc_all[-1] > best_acc:
            #保存当前最高准确度
            best_acc = val_acc_all[-1]
            #保存当前最好参数
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time() - since
        print('训练和验证耗费的时间{:.0f}m{:.0f}s'.format(time_use//60,time_use%60))

    #选择最优模型
    #加载最高准确率下的模型参数

    torch.save(best_model_wts, 'D:/py_project/LeNet5/best_model.pth')

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))


    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")


    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #将模型实例化
    LeNet5 = LeNet5()
    train_lodar, val_lodar = train_val_data_process()
    train_process = train_model_process(LeNet5,train_lodar,val_lodar,50)
    matplot_acc_loss(train_process)
