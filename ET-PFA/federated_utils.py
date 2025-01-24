import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import copy
import utils
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
import math
import torch.optim as optim
from PIL import Image
import torch.nn as nn


def federated_setup(global_model, args):
    if args.data == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                    ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    elif args.data == 'cifar10':
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        # test_loader = torch.utils.data.DataLoader(
        #     datasets.CIFAR10('./data', train=False, split='byclass', transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((args.norm_mean,), (args.norm_std,))
        #     ])),
        #     batch_size=args.test_batch_size, shuffle=False)
        test_data = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ]))

        classes = train_data.classes
        n_classes = len(classes)

        labels = np.concatenate(
            [np.array(train_data.targets), np.array(test_data.targets)], axis=0)
        dataset = ConcatDataset([train_data, test_data])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 设置数据文件夹路径和标签文件路径
        data_dir_1 = './data/dataset1'
        label_file_1 = './data/labels1.csv'
        #
        # data_dir_2 = './data/data2-1'
        # label_file_2 = './data/label2-1.csv'
        #
        # data_dir_3 = './data/data3'
        # label_file_3 = './data/label3_new.csv'

        # 创建数据集对象
        dataset = CustomDataset(data_dir_1, label_file_1, transform=transform)
        # 计算训练集和测试集的大小
        torch.manual_seed(42)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # # 划分训练集和测试集
        # train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
        #
        # train_data.classes = dataset.classes
        #
        # # labels = dataset.labels
        # # train_data.dataset.targets = labels[train_data.indices]
        labels = dataset.labels.values[:, 0]
        #
        # # 创建数据加载器
        # # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        # # test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

        classes = dataset.classes
        n_classes = len(classes)


    # 狄利克雷分布
    # labels = train_data.classes
    # labels = np.array(dataset.targets)
    # client_idcs = utils.dirichlet_split_noniid(train_data, args)
    client_idcs = utils.dirichlet_split_noniid(
        train_labels=labels, alpha=args.alpha, n_clients=args.num_users, args=args)
    # 画图
    # n_classes = 10
    n_classes = len(classes)
    n_clients = args.num_users

    arr = np.array([0, 1, 2, 3, 4])

    # 展示不同client上的label分布
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    # label_distribution = [[] for _ in range(5)]
    # label_distribution = [[] for _ in range(labels)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    # plt.hist(label_distribution, stacked=True,
    #          bins=np.arange(-0.5, n_clients + 1.5, 1),
    #          label=labels, rwidth=0.5)
    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=arr, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                      c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig(f'fig/{args.data}_K={args.num_users}_Non-iid_alpha={args.alpha}.png')
    # plt.show()

    indexes = torch.randperm(len(dataset))
    user_data_len = math.floor(len(dataset) / args.num_users)
    local_models = {}
    local_data_sizes = []

    for user_idx in range(args.num_users):

        user_data = torch.utils.data.Subset(dataset, client_idcs[user_idx])
        train_len = round(len(user_data) * 0.8)
        local_data_sizes.append(train_len)
        train_data, test_data = torch.utils.data.random_split(user_data, [train_len, len(user_data) - train_len],
                                                              generator=torch.Generator().manual_seed(0))

        # train_data, val_data = torch.utils.data.random_split(user_data, [len(user_data) - amount, amount])
        val_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        user = {'train_loader': train_loader,
                'test_loader': val_loader,
                'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models, local_data_sizes


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.labels = pd.read_csv(label_file, index_col=0)  # 使用第一列作为索引
        self.transform = transform
        self.classes = self._get_classes()

    def _get_classes(self):
        # 获取标签列并去重，生成类别列表
        # unique_labels = self.labels.unique()
        unique_labels = np.unique(self.labels)
        return sorted(unique_labels.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name = self.labels.index[index]  # 使用索引获取图像文件名
        image_path = os.path.join(self.data_dir, image_name)
        # image = Image.open(image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(self.labels.iloc[index, 0], dtype=torch.long)  # 获取对应的标签
        flag = torch.tensor(0.)
        if len(self.labels.iloc[index]) > 1:
            flag = torch.tensor(self.labels.iloc[index, 1], dtype=torch.long)  # 获取对应的标签

        # return image, label, flag
        return image, label


def distribute_model_feature(local_models, global_model):
    global_features_state = global_model.features.state_dict()
    local_features_states = {}  # 用于存储每个用户的 features state_dict

    for user_idx in range(len(local_models)):
        local_model = local_models[user_idx]['model']
        local_feature_states = local_model.features.state_dict()
        local_model.features.load_state_dict(copy.deepcopy(global_model.features.state_dict()))
        local_feature_states_1 = local_model.features.state_dict()

        # 保存每个用户模型的 features 的 state_dict
        local_features_states[f"local_features_{user_idx + 1}"] = copy.deepcopy(local_feature_states_1)


def distribute_model_features(local_models, global_model, client_updates):
    local_features_states = {}  # 用于存储每个用户的 features state_dict

    for user_idx in range(len(local_models)):
        if any(update['idx'] == user_idx for update in client_updates):
            local_model = local_models[user_idx]['model']
            local_model.features.load_state_dict(copy.deepcopy(global_model.features.state_dict()))
            local_feature_states_1 = local_model.features.state_dict()

            # 保存每个用户模型的 features 的 state_dict
            local_features_states[f"local_features_{user_idx + 1}"] = copy.deepcopy(local_feature_states_1)


def distribute_model(local_models, global_model):
    g_s = global_model.state_dict()
    l_s = []

    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))

        l_s.append(local_models[user_idx]['model'].state_dict())

    return g_s, l_s

# def aggregate_models(local_models, global_model):  # FeaAvg
#     state_dict = copy.deepcopy(global_model.state_dict())
#     for key in state_dict.keys():
#         local_weights_average = torch.zeros_like(state_dict[key])
#         for user_idx in range(0, len(local_models)):
#             local_weights_orig = local_models[user_idx]['model'].features.state_dict()[key] - state_dict[key]
#             local_weights_average += local_weights_orig
#         state_dict[key] += (local_weights_average / len(local_models)).to(state_dict[key].dtype)
#     global_model.load_state_dict(copy.deepcopy(state_dict))


def aggregate_models(local_models, global_model, local_data_sizes):  # FeaAvg
    # 获取全局模型的状态字典
    state_dict = copy.deepcopy(global_model.state_dict())

    # 计算所有参与方的数据总数
    total_data_size = sum(local_data_sizes)

    # 对全局模型的每个参数进行聚合
    for key in state_dict.keys():
        # 初始化参数的聚合结果为零
        aggregated_weights = torch.zeros_like(state_dict[key])

        #
        for user_idx in range(len(local_models)):
            local_state_dict = local_models[user_idx]['model'].features.state_dict()
            # local_state_dict = local_models[user_idx]['model'].classifier.state_dict()
            # 加权累加，权重为每个参与方数据量的比例
            aggregated_weights += (local_state_dict[key] * local_data_sizes[user_idx] / total_data_size)

        # 更新全局模型的参数为聚合结果
        state_dict[key] = aggregated_weights.to(state_dict[key].dtype)

    # 将更新后的参数加载到全局模型中
    global_model.load_state_dict(copy.deepcopy(state_dict))


def federated_averaging(global_params, client_updates):
    total_clients = len(client_updates)
    for param in global_params.keys():
        global_params[param] = sum([client[param] for client in client_updates]) / total_clients
    return global_params


def federated_learning(clients, global_model, test_loader, num_epochs=50, alpha=0.001, beta=1.2, local_epochs=5,
                       learning_rate=0.001):
    global_params = global_model.state_dict()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        client_updates = []

        for client in clients:
            # 每个客户端判断是否加载全局参数
            if 'local_params' not in client:
                client['model'].load_state_dict(global_params)
            else:
                client['model'].load_state_dict(client['local_params'])

            loss_pre, _ = evaluate(client['model'], client['val_loader'], client['criterion'])
            local_params_before = copy.deepcopy(client['model'].state_dict())

            optimizer = optim.SGD(client['model'].parameters(), lr=client['lr'], momentum=0.9)
            client['model'] = local_training(client['model'], client['train_loader'], optimizer, client['criterion'],
                                             epochs=local_epochs)

            local_params_after = copy.deepcopy(client['model'].state_dict())
            loss_post, _ = evaluate(client['model'], client['val_loader'], client['criterion'])

            if event_trigger(local_params_before, local_params_after, loss_pre, loss_post, alpha, beta, epoch,
                             learning_rate):
                print(f"Client {client['id']} triggered communication.")
                client_updates.append(local_params_after)
                client.pop('local_params', None)
            else:
                client['local_params'] = local_params_after

        if client_updates:
            global_params = federated_averaging(global_params, client_updates)

        for client in clients:
            found_id = False
            for c in client_updates:
                if 'id' in c and client['id'] == c['id']:
                    client['model'].load_state_dict(global_params)
                    found_id = True
                    break
            if not found_id:
                client['model'].load_state_dict(client.get('local_params', global_params))

        global_model.load_state_dict(global_params)
        test_loss, test_accuracy = evaluate(global_model, client['test_loader'], nn.CrossEntropyLoss())
        print(f"Global Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    loss /= len(data_loader)
    return loss, accuracy

def local_training(model, train_loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model

# 事件触发通信条件
def event_trigger(local_params_before, local_params_after, loss_pre, loss_post, alpha, beta, epoch, learning_rate):
    param_diff = sum(torch.norm(local_params_after[key] - local_params_before[key], p=2) for key in local_params_before)
    loss_diff = loss_post - loss_pre
    Omega = sum(param.numel() for param in local_params_before.values())

    A = (param_diff / (Omega ** 0.5)) * loss_diff
    B = (alpha * learning_rate) / ((epoch + 1) ** beta)

    print(f"A:{A}, B:{B}")

    return A >= B


# 联邦平均
def federated_averaging(global_params, client_updates):
    total_clients = len(client_updates)
    for param in global_params.keys():
        global_params[param] = sum([client.features[param] for client in client_updates]) / total_clients
    return global_params