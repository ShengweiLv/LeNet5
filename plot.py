from torchvision.datasets import FashionMNIST
import numpy as np
import torch.utils.data as data
from torchvision import transforms  #处理数据集
import matplotlib.pyplot as plt



train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()]),
                          download=True)


#捆绑数据,64一组，打乱数据
train_loader = data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)

# plt.figure(figsize=(12, 5))
# for ii in np.arange(len(batch_y)):
#     plt.subplot(4,16,ii+1)
#     plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
#     plt.title(class_label[batch_y[ii]],size=10)
#     plt.axis('off')
#     plt.subplots_adjust(wspace=0.05)
# plt.show()