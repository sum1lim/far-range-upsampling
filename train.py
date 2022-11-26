#!/usr/bin/env python3
import argparse
import torch
import sys
import numpy as np
import time
import torch.optim as optim
import sklearn.metrics as metrics
from torch import nn
import torch.nn.functional as F
from models import model1
from utils import LidarData, MSIE_Loss, Focal_Loss, combined_Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


model_dict = {"model1": model1}
loss_dict = {"msie": MSIE_Loss, "focal": Focal_Loss, "combined": combined_Loss}


def main(args):
    log_file = open(f"./checkpoints/{args.exp_name}_train.log", "w")

    device = torch.device("cuda")
    BS = torch.cuda.device_count() * 8

    print(f"Number of GPUs: {torch.cuda.device_count()}", file=sys.stdout)
    print(f"Number of GPUs: {torch.cuda.device_count()}", file=log_file)
    print(f"Batch size: {BS}", file=sys.stdout)
    print(f"Batch size: {BS}", file=log_file)

    tr_data_loader = DataLoader(
        LidarData("tr", step_size=args.KNNstep),
        num_workers=8,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
    )
    te_data_loader = DataLoader(
        LidarData("te", step_size=args.KNNstep),
        num_workers=8,
        batch_size=BS,
        shuffle=True,
        drop_last=True,
    )

    model = nn.DataParallel(model_dict[args.model](batch_size=BS, device=device)).to(
        device
    )

    epochs = 10000
    loss_func = loss_dict[args.loss]().to(device)
    learning_rate = 0.001
    optim_net = optim.Adam(model.parameters(), lr = learning_rate)
    best_test_loss = np.inf

    patience = 0
    patience_count = 0
    for epoch in range(epochs):
        train_loss = 0.0
        count = 0.0  # numbers of data
        train_pred = []
        train_true = []
        idx = 0  # iterations
        total_time = 0.0

        for point, data, label in tr_data_loader:
            point, data, label = point.to(device), data.to(device), label.to(device)

            start_time = time.time()

            logits = model(point, data)

            optim_net.zero_grad()
            loss = loss_func(logits, label)
            loss.backward(retain_graph=True)
            optim_net.step()
            end_time = time.time()
            total_time += end_time - start_time

            gt = label.flatten(start_dim=0, end_dim=1)
            preds = logits.flatten(start_dim=0, end_dim=1)
            count += BS
            train_loss += loss.item() * BS
            train_true.append(gt.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            idx += 1

        print("train total time is", total_time, file=sys.stdout)
        print("train total time is", total_time, file=log_file)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        train_pred_class = np.absolute(train_pred)
        train_pred_class[train_pred_class < 1000] = 1
        train_pred_class[train_pred_class >= 1000] = 0
        train_true_class = np.absolute(train_true)
        train_true_class[train_true_class < 1000] = 1
        train_true_class[train_true_class >= 1000] = 0
        try:
            train_mae = metrics.mean_absolute_error(train_true, train_pred)
            train_acc = metrics.balanced_accuracy_score(
                train_true_class, train_pred_class
            )
        except ValueError:
            train_mae = "NaN"
            train_acc = "NaN"
        outstr = f"Epoch {epoch}, train MAE: {train_mae}, train accuracy: {train_acc} loss: {train_loss * 1.0 / count}"
        print(outstr, file=sys.stdout)
        print(outstr, file=log_file)

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        total_time = 0.0
        for point, data, label in te_data_loader:
            point, data, label = point.to(device), data.to(device), label.to(device)
            start_time = time.time()
            logits = model(point, data)
            end_time = time.time()
            total_time += end_time - start_time
            loss = loss_func(logits, label)
            gt = label.flatten(start_dim=0, end_dim=1)
            preds = logits.flatten(start_dim=0, end_dim=1)
            count += BS
            test_loss += loss.item() * BS
            test_true.append(gt.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        print("test total time is", total_time, file=sys.stdout)
        print("test total time is", total_time, file=log_file)
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_pred_class = np.absolute(test_pred)
        test_pred_class[test_pred_class < 1000] = 1
        test_pred_class[test_pred_class >= 1000] = 0
        test_true_class = np.absolute(test_true)
        test_true_class[test_true_class < 1000] = 1
        test_true_class[test_true_class >= 1000] = 0
        try:
            test_mae = metrics.mean_absolute_error(test_true, test_pred)
            test_acc = metrics.balanced_accuracy_score(test_true_class, test_pred_class)
        except ValueError:
            test_mae = "NaN"
            test_acc = "NaN"
        outstr = f"Epoch {epoch}, test MAE: {test_mae}, test accuracy: {test_acc}, loss: {test_loss * 1.0 / count}"
        print(outstr, file=sys.stdout)
        print(outstr, file=log_file)

        if test_loss < best_test_loss:
            patience = 0
            best_test_loss = test_loss
            print("save checkpoint", file=sys.stdout)
            print("save checkpoint", file=log_file)
            torch.save(model.state_dict(), f"checkpoints/{args.exp_name}_model.pt")

        patience += 1
        if patience > 50:
            if patience_count < 3:
                patience = 0
                patience_count += 1
                learning_rate /= 10
                print("learning rate change", file=sys.stdout)
                print("learning rate change", file=log_file)
                optim_net = optim.Adam(model.parameters(), lr=learning_rate)
            else:
                print("early stopping", file=sys.stdout)
                print("early stopping", file=log_file)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--KNNstep",
        type=int,
        default=8,
        help="KNN step size",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=model_dict.keys(),
        help="Model to use for training",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=loss_dict.keys(),
        help="Loss function",
    )

    args = parser.parse_args()

    main(args)
