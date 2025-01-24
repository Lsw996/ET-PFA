import os
from statistics import mean
import torch
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import torch_pruning as tp


def data(args):
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
    else:
        train_data = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((args.norm_mean,), (args.norm_std,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((args.norm_mean,), (args.norm_std,))
            ])),
            batch_size=args.test_batch_size, shuffle=False)
    return train_data, test_loader


def data_split(data, amount, args):
    # split train, validation
    train_data, val_data = torch.utils.data.random_split(data, [len(data) - amount, amount])
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False)

    # input, output sizes
    in_channels, dim1, dim2 = data[0][0].shape  # images are dim1 x dim2 pixels
    input = dim1 * dim2 if args.model == 'mlp' or args.model == 'linear' else in_channels
    output = len(data.classes)  # number of classes

    return input, output, train_data, val_loader


def train_one_epoch(train_loader, model,
                    optimizer, creterion,
                    device):
    state = model.features.state_dict()
    model.train()
    losses = []
    # if iterations is not None:
    #     local_iteration = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # send to device
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = creterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # if iterations is not None:
        #     local_iteration += 1
        #     if local_iteration == iterations:
        #         break
    return mean(losses)


def test(test_loader, model, creterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 禁用梯度计算
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)  # send to device

            output = model(data)
            test_loss += creterion(output, label).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss


def initializations(args):
    #  reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    #  documentation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')

    best_val_acc = np.NINF
    path_best_model = 'checkpoints/' + args.exp_name + '/model.best.t7'

    return boardio, textio, best_val_acc, path_best_model


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def dirichlet_split_noniid(train_labels, alpha, n_clients, args):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    np.random.seed(args.seed)
    n_classes = train_labels.max()+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes, )
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def pruning_model(model, args):

    # 1. Importance criterion  重要性标准
    # imp = tp.importance.GroupTaylorImportance()  # or GroupNormImportance(p=2), GroupHessianImportance(), etc.
    imp = tp.importance.MagnitudeImportance(p=1)

    # 2. Initialize a pruner with the model and the importance criterion  将模型和重要性评估标准一起用来初始化一个剪枝器。
    # ignored_layers = [model.conv1, model.fc]
    example_inputs = torch.randn(3, 32, 32)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features <= 16:
            ignored_layers.append(m)
        # if isinstance(m, torch.nn.Conv2d) and m.out_channels <= 16:
        #     ignored_layers.append(m)

    example_inputs = example_inputs.to(args.device)

    pruner = tp.pruner.MagnitudePruner(  # We can always choose MetaPruner if sparse training is not required.
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=args.pl, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        # ch_sparsity_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized sparsity for layers or blocks
        ignored_layers=ignored_layers,
    )

    # # 3. Prune & finetune the model

    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # if isinstance(imp, tp.importance.GroupTaylorImportance):
    #     # Taylor expansion requires gradients for importance estimation 泰勒展开需要梯度来进行重要性估计
    #     # A dummy loss, please replace it with your loss function and data!
    #     output = model(example_inputs)
    #     loss = creterion(output, example_label)
    #     # loss = model(example_inputs).sum()
    #     loss.backward()  # before pruner.step()

    pruner.step()

    return model