import argparse

import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algorithms', type=str, default='DETFL',
                        choices=['ET-PFL', 'DSGD', 'ETFed', 'DETFL'])

    parser.add_argument('--exp_name', type=str, default='exp',
                        help="the name of the current experiment")
    parser.add_argument('--eval', default=False,
                        help="weather to perform inference of training")

    # data arguments
    parser.add_argument('--data', type=str, default='pd',
                        choices=['mnist', 'cifar10', 'pd'],
                        help="dataset to use (mnist or cifar)")
    parser.add_argument('--alpha', type=float, default=100,       # 0.5
                        help="Dirichlet para")
    parser.add_argument('--norm_mean', type=float, default=0.5,
                        help="normalize the data to norm_mean")
    parser.add_argument('--norm_std', type=float, default=0.5,
                        help="normalize the data to norm_std")
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help="trainset batch size")
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help="testset batch size")

    # federated arguments
    parser.add_argument('--model', type=str, default='AlexNet',
                        choices=['cnn2', 'cnn3', 'LeNet5_H', 'linear', 'CNN3_H', 'AlexNet'],
                        help="model to use (cnn, mlp)")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users participating in the federated learning")
    parser.add_argument('--global_epochs', type=int, default=50,
                        help="number of global epochs")

    # acc trigger
    parser.add_argument('--local_epochs', type=int, default=5,
                        help="number of local epochs")

    # learning arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help="optimizer to use (sgd or adam)")
    parser.add_argument('--lr', type=float, default=0.00001,
                        help="learning rate is 0.01 for cnn and 0.1  for linear")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="momentum")
    parser.add_argument('--lr_scheduler', default=True,
                        help="reduce the learning rat when val_acc has stopped improving (increasing)")
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cpu'],
                        help="device to use (gpu or cpu)")
    parser.add_argument('--seed', type=float, default=0,
                        help="manual seed for reproducibility")

    args = parser.parse_args()
    return args

