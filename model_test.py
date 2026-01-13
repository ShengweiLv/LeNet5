import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet5

def test_data_process():#处理训练集和验证集
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)


    test_loader = Data.DataLoader(dataset=test_data,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=0)

    return test_loader

def test_model_process(model, test_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    #初始化参数
    test_acc = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            pre_label = torch.argmax(output, dim = 1)

            test_acc += torch.sum(pre_label == test_data_y.data)
            test_num += test_data_x.size(0)
    #计算准确率
    test_accs = test_acc.double().item() / test_num

    print("测试的准确率为：",test_accs)

if __name__ == "__main__":
    model = LeNet5()

    #模型序列化
    model.load_state_dict(torch.load("best_model.pth"))

    test_loader = test_data_process()
    test_model_process(model, test_loader)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    #
    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x, b_y in test_loader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         #设置模型为验证模式
    #         model.eval()
    #         output = model(b_x)
    #         pre_label = torch.argmax(output, dim=1) #张量的模式
    #         result = pre_label.item()
    #         label = b_y.item()
    #
    #         print("预测值：",classes[result],"-----------","真实值：",classes[label])