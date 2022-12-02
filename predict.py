#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import LidarData
from collections import OrderedDict
from models import *

model_dict = {
    "model0_0": model0_0,
    "model0_1": model0_1,
    "model1": model1,
    "model2": model2,
}


def main(args):
    BS = 1
    lidar_input = np.loadtxt(args.input, delimiter=",", dtype=np.double)
    lidar_input = lidar_input[lidar_input[:, 0] > 0.2]

    device = torch.device("cpu")
    model = model_dict[args.model](batch_size=BS, device=device)
    state_dict = torch.load(args.params, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.double()

    samples = np.random.rand(512 * args.num_generation, 3)
    samples[:, 0] = samples[:, 0] * 150000 + 95000
    samples[:, 1] = (samples[:, 1] - 0.5) * 100000
    samples[:, 2] = (samples[:, 2] - 0.5) * 30000
    samples /= 245000
    points = torch.tensor(np.array([samples], dtype=np.double)).to(device)

    KNN_data = np.zeros((512 * args.num_generation, 128 * 3))

    for i in range(512 * args.num_generation):
        # KNN
        distances = np.linalg.norm((lidar_input - samples[i]), axis=1)
        KNN_indices = np.argsort(distances)[:128]
        KNN_data[i] = (lidar_input[KNN_indices] - samples[i]).flatten(order="C")

    for i in range(args.num_generation):
        chunk_data = KNN_data[i * 512 : i * 512 + 512]
        chunk_points = points[:, i * 512 : i * 512 + 512]

        chunk_data = chunk_data.reshape((chunk_data.shape[0], 128, 3))
        chunk_data = chunk_data[:, [i * args.KNNstep for i in range(16)]]
        data = torch.tensor(np.array([chunk_data], dtype=np.double)).to(device)

        prediction = (
            model(chunk_points.double(), data.double()).cpu().detach().numpy()[0]
        )
        if i == 0:
            prediction_cat = prediction
        else:
            prediction_cat = np.r_["0", prediction_cat, prediction]

    output = np.c_[samples, prediction_cat]
    output = output[output[:, -1] < args.threshold]

    np.savetxt(
        f"./output/{args.input.split('/')[-1].replace('visible.txt', '_output')}.csv",
        output,
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
        choices=model_dict.keys(),
        help="Model to use for training",
    )
    parser.add_argument(
        "--params",
        type=str,
        help="Path to the saved parameters for the model",
    )
    parser.add_argument(
        "--threshold",
        default=np.inf,
        type=int,
        help="Threshold value of the distance output",
    )
    parser.add_argument(
        "--num-generation",
        default=4,
        type=int,
        help="Number of output generations",
    )

    args = parser.parse_args()

    main(args)
