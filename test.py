#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from models import model_dict
from utils import LidarData, loss_dict
from collections import OrderedDict
from models import *


def main(args):
    te_data_loader = DataLoader(
        LidarData("te", step_size=args.KNNstep),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda")
    model = model_dict[args.model](batch_size=args.batch_size, device=device)
    state_dict = torch.load(args.params, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.double().to(device)

    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    for point, data, label in te_data_loader:
        point, data, label = point.to(device), data.to(device), label.to(device)
        logits = model(point.double(), data.double())
        gt = label.flatten(start_dim=0, end_dim=1)
        preds = logits.flatten(start_dim=0, end_dim=1)
        count += args.batch_size
        test_true.append(gt.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_pred_class = np.absolute(test_pred)
    test_pred_class[test_pred_class < 1000] = 1
    test_pred_class[test_pred_class >= 1000] = 0
    test_true_class = np.absolute(test_true)
    test_true_class[test_true_class < 1000] = 1
    test_true_class[test_true_class >= 1000] = 0

    test_mae = metrics.mean_absolute_error(test_true, test_pred)
    test_acc = metrics.balanced_accuracy_score(test_true_class, test_pred_class)

    print(
        f"""
            Test MAE: {test_mae}\n
            Test accuracy: {test_acc}
        """
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--KNNstep",
        type=int,
        default=1,
        help="KNN step size",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=model_dict.keys(),
        help="Model to use for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Path to the saved parameters for the model",
    )

    args = parser.parse_args()

    main(args)