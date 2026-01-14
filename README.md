#基于LeNet5对FashionMNIST进行识别

##FashionMNIST数据集介绍
    样本数量：
        训练集：60,000 张图片
        测试集：10,000 张图片
    图片尺寸：28 x 28 像素（单通道灰度图，Grayscale）。
    类别数量：10 个类别。
    FashionMNIST 包含 10 种日常服装和配饰，每个类别对应一个数字标签（0-9）：
    标签 (Label)	类别 (Class)	中文名称	样例描述
    0	T-shirt/top	T 恤 / 上衣	短袖、长袖上衣
    1	Trouser	裤子	长裤
    2	Pullover	套头衫	毛衣、卫衣（无拉链）
    3	Dress	连衣裙	裙子
    4	Coat	外套	夹克、风衣（有拉链）
    5	Sandal	凉鞋	露趾鞋
    6	Shirt	衬衫	翻领衬衫
    7	Sneaker	运动鞋	球鞋、板鞋
    8	Bag	包	手提包
    9	Ankle boot	踝靴	高帮鞋、靴子
##项目介绍
通过构建LeNet5网络对FashionMINST进行训练，得到预测FashionMNIST较好的参数（形成best_model.pth文件），并可以在测试集进行测试，精度可以达到88%左右，网络不深，但效果较好。

##如何使用
1、运行model_train.py文件进行训练，生成best_model.pth，然后可以使用model_test.py进行预测。
