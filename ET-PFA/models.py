import torch.nn
import torch.nn as nn
import torch.nn.functional as F


class Linear(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.linear(x)
        return out


class FC2Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FC2Layer, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CNN2Layer(nn.Module):
    def __init__(self):
        super(CNN2Layer, self).__init__()
        self.n_feature = 6
        self.intemidiate_size = 4
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=5)
        self.fc1 = nn.Linear(6 * self.intemidiate_size * self.intemidiate_size, 50)  # 4*4 for MNIST 5*5 for CIFAR10
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, self.n_feature * self.intemidiate_size * self.intemidiate_size)  # 4*4 for MNIST 5*5 for CIFAR10
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class CNN3Layer(nn.Module):
    def __init__(self):
        super(CNN3Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout1 = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.conv3 = nn.Conv2d(64, 128, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout(p=0.2, inplace=False)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.dropout1(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.dropout1(x)
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.dropout1(x)
#         return x
#
#
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(128 * 2 * 2, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 10)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
#
#
# class CNN3_H(nn.Module):
#     def __init__(self):
#         super(CNN3_H, self).__init__()
#         self.feature = FeatureExtractor()
#         self.classifier = Classifier()
#
#     def forward(self, x):
#         x = self.feature(x)
#         x = x.view(-1, 128 * 2 * 2)
#         x = self.classifier(x)
#         return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, (2, 2))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, (2, 2))
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):        # 普通Block简单完成两次卷积操作
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = x                                            # 普通Block的shortcut为直连，不需要升维下采样

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)       # 完成一次卷积
        x = self.bn2(self.conv2(x))                             # 第二次卷积不加relu激活函数

        x += identity                                           # 两路相加
        return F.relu(x, inplace=True)                          # 添加激活函数输出


class SpecialBlock(nn.Module):                                  # 特殊Block完成两次卷积操作，以及一次升维下采样
    def __init__(self, in_channel, out_channel, stride):        # 注意这里的stride传入一个数组，shortcut和残差部分stride不同
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(                    # 负责升维下采样的卷积网络change_channel
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)                       # 调用change_channel对输入修改，为后面相加做变换准备

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))                             # 完成残差部分的卷积

        x += identity
        return F.relu(x, inplace=True)                          # 输出卷积单元


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # self.prepare = nn.Sequential(           # 所有的ResNet共有的预处理==》[batch, 64, 56, 56]
        #     nn.Conv2d(3, 64, 7, 2, 3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, 2, 1)
        # )
        # self.layer1 = nn.Sequential(            # layer1有点特别，由于输入输出的channel均是64，故两个CommonBlock
        #     CommonBlock(64, 64, 1),
        #     CommonBlock(64, 64, 1)
        # )
        # self.layer2 = nn.Sequential(            # layer234类似，由于输入输出的channel不同，故一个SpecialBlock，一个CommonBlock
        #     SpecialBlock(64, 128, [2, 1]),
        #     CommonBlock(128, 128, 1)
        # )
        # self.layer3 = nn.Sequential(
        #     SpecialBlock(128, 256, [2, 1]),
        #     CommonBlock(256, 256, 1)
        # )
        # self.layer4 = nn.Sequential(
        #     SpecialBlock(256, 512, [2, 1]),
        #     CommonBlock(512, 512, 1)
        # )
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = nn.Sequential(                # 最后用于分类的全连接层，根据需要灵活变化
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)         # 这个使用CIFAR10数据集，定为10分类
        )

    def forward(self, x):

        x = self.feature(x)          # 四个卷积单元
        x = x.reshape(x.shape[0], -1)   # 将x展平，输入全连接层
        x = self.classifier(x)

        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_checkpoint(model: torch.nn.Module, path: str, device):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

