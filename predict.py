#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from models import model1
from torch.utils.data import DataLoader
from utils import LidarData
from collections import OrderedDict


def main(args):
    BS = 1
    lidar_input = np.loadtxt(args.input, delimiter=",", dtype=np.double)

    device = torch.device("cpu")
    model = model1(batch_size=BS)
    state_dict = torch.load(args.model, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.double()

    samples = np.random.rand(1024, 3)
    samples[:, 0] = samples[:, 0] * 150000 + 95000
    samples[:, 1] = (samples[:, 1] - 0.5) * 100000
    samples[:, 2] = (samples[:, 2] - 0.5) * 30000
    samples /= 245000
    points = torch.tensor(np.array([samples], dtype=np.double)).to(device)

    KNN_data = np.zeros((1024, 128 * 4))

    for i in range(1024):
        # KNN
        distances = np.linalg.norm((lidar_input - samples[i]), axis=1)
        KNN_indices = np.argsort(distances)[:128]
        KNN_data[i] = np.c_[lidar_input[KNN_indices], distances[KNN_indices]].flatten(
            order="C"
        )

    KNN_data = KNN_data.reshape((KNN_data.shape[0], 128, 4))
    KNN_data = KNN_data[:, [i * args.KNNstep for i in range(16)]]
    data = torch.tensor(np.array([KNN_data], dtype=np.double)).to(device)

    mask1 = torch.ones(1024 * BS, 16).bool().to(device)
    mask2 = torch.ones(BS, 1024).bool().to(device)

    prediction = model(points.double(), data.double(), mask1, mask2)
    print(prediction)

    np.savetxt(
        f"./output/{args.input.split('/')[-1]}.csv",
        np.c_[samples, prediction.cpu().detach().numpy()[0]],
        delimiter=",",
        fmt="%f",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        help="Input LIDAR scene",
    )
    parser.add_argument(
        "--KNNstep",
        type=int,
        help="KNN step size",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use for prediction",
    )

    args = parser.parse_args()

    main(args)
