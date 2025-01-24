import gc
import math
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import utils
import models
import federated_utils
from configurations import args_parser
from torchinfo import summary
from statistics import mean
import copy

if __name__ == '__main__':

    RED_TEXT = "\033[91m"
    RESET_TEXT = "\033[0m"

    GREEN_TEXT = "\033[1;32m"

    args = args_parser()

    start_time = time.time()

    boardio, textio, _, _ = utils.initializations(args)
    textio.cprint(str(args))

    # 模型
    if args.model == 'cnn2':
        global_model = models.CNN2Layer()
    elif args.model == 'ResNet18':
        global_model = models.ResNet18()
    elif args.model == 'AlexNet':
        global_model = models.AlexNet()
        global_model = models.load_checkpoint(global_model, 'model/alexnet-owt-4df8aa71.pth', args.device)
        num_features = global_model.classifier[6].in_features
        global_model.classifier[6] = torch.nn.Linear(num_features, 5)

    model_summary = summary(global_model)
    Params = model_summary.total_params
    global_model.to(args.device)

    train_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    train_loss_list, val_acc_list = [], []

    local_models, local_data_sizes = federated_utils.federated_setup(global_model, args)
    initial_acc, total_epochs, total_coms = [0] * args.num_users, [0] * args.num_users, [0] * args.num_users

    output_history = []

    if args.algorithms == 'ET-PFL':
        for global_epoch in tqdm(range(args.global_epochs)):
            local_data_size = []
            if global_epoch == 0:
                federated_utils.distribute_model(local_models, global_model)
            else:
                federated_utils.distribute_model_feature(local_models, global_model)

            client_updates = []
            users_loss, users_acc = [], []

            for user_idx in range(args.num_users):
                communication = 0
                user = local_models[user_idx]
                user['idx'] = user_idx
                no_improvement_count, user_loss = 0, []

                for local_epoch in range(args.local_epochs):
                    train_loss = utils.train_one_epoch(user['train_loader'], user['model'], user['opt'],
                                                       train_criterion, args.device)
                    user_loss.append(train_loss)
                    if args.lr_scheduler:
                        user['scheduler'].step(train_loss)

                    current_acc, _ = utils.test(user['test_loader'], user['model'], test_criterion, args.device)
                    acc_diff = current_acc - initial_acc[user_idx]
                    print(
                        f'user: {user_idx + 1} | epoch: {local_epoch + 1} | current_initial_acc: {initial_acc[user_idx]:.2f}%'
                        f' | current_acc: {current_acc:.2f}% | acc_diff: {acc_diff:.2f}%')

                    at = (100 - current_acc) / 2
                    n_i_e = math.ceil(args.local_epochs / 2)
                    if acc_diff >= at:
                        no_improvement_count = 0
                        communication = 1
                    elif acc_diff == 0:
                        no_improvement_count += 1

                    if no_improvement_count >= n_i_e or current_acc == 100:
                        break

                    total_epochs[user_idx] += 1

                if communication:
                    print(f"Client {user_idx + 1} triggered communication.")
                    client_updates.append(user)
                    total_coms[user_idx] += 1

                users_loss.append(mean(user_loss))
                users_acc.append(current_acc)

            train_loss = mean(users_loss)
            val_acc = mean(users_acc)

            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)

            if global_epoch < args.global_epochs - 1:
                for user in client_updates:
                    train_len = len(user['train_loader'].dataset)
                    local_data_size.append(train_len)
                federated_utils.aggregate_models(client_updates, global_model.features, local_data_size)
            else:
                for user_idx in range(args.num_users):
                    user_model_path = f'model/user_{user_idx + 1}_final_model.pth'
                    torch.save(local_models[user_idx]['model'].state_dict(), user_model_path)

            boardio.add_scalar('train', train_loss_list[-1], global_epoch)
            boardio.add_scalar('validation', val_acc_list[-1], global_epoch)
            gc.collect()

            output_message = f'{GREEN_TEXT}epoch: {global_epoch + 1} | train_loss: {train_loss:.2f} | ' \
                             f'val_acc: {val_acc:.2f} | train_epoch:{sum(total_epochs)} | ' \
                             f'communications:{sum(total_coms)}{RESET_TEXT}'
            textio.cprint(output_message)

            for user_idx in range(args.num_users):
                print(f'user: {user_idx} | total_epochs: {total_epochs[user_idx]} | total_coms:{total_coms[user_idx]}')

            initial_acc = users_acc.copy()

    for user_idx in range(args.num_users):
        textio.cprint(f'user: {user_idx} | total_epochs: {total_epochs[user_idx]} | total_coms:{total_coms[user_idx]}')

    textio.cprint(f'Total epochs across all users: {sum(total_epochs)}')
    textio.cprint(f'Total communications across all users: {sum(total_coms)}')
    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)
    textio.cprint(f'total execution time: {(time.time() - start_time) / 60:.0f} min')
