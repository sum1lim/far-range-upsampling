#!/usr/bin/env python3
import argparse
import torch
import time
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
    BS = 4
    lidar_input = np.loadtxt(args.input, delimiter=",", dtype=np.double)
    lidar_input = lidar_input[lidar_input[:, 0] > 0.25]

    device = torch.device("cuda")
    model = model_dict[args.model](batch_size=BS, device=device)
    state_dict = torch.load(args.params, map_location=device)

    total_time = 0
    input_start_time = time.time()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.double().to(device)

    samples = np.random.rand(512 * args.num_generation, 3)
    samples[:, 0] = samples[:, 0] * 150000 + 95000
    samples[:, 1] = (samples[:, 1] - 0.5) * 100000
    samples[:, 2] = (samples[:, 2] - 0.5) * 30000
    samples /= 245000
    points = torch.tensor(np.array([samples], dtype=np.double)).to(device)

    lidar_input = torch.tensor(lidar_input).unsqueeze(0).to(device)
    if points.shape[1] < lidar_input.shape[1]:
        points = torch.cat(
            (
                points,
                torch.zeros((1, lidar_input.shape[1] - points.shape[1], 3)).to(device),
            ),
            1,
        )
    rel_dist = (points[:, :, None, :] - lidar_input[:, None, :, :])[
        :, : samples.shape[0]
    ].norm(dim=-1)
    dist, indices = rel_dist.topk(16 * args.KNNstep, largest=False)
    KNN_data = lidar_input[0][indices[0]] - points[0].unsqueeze(1).repeat([1, 16, 1])

    KNN_data = torch.stack(KNN_data.split(512, dim=0))
    points = torch.stack(points[0].split(512, dim=0))

    input_end_time = time.time()
    print(f"Input processing time: {input_end_time - input_start_time}")
    total_time += input_end_time - input_start_time

    for i in range(args.num_generation // BS):
        batch_start_time = time.time()
        chunk_data = KNN_data[i * BS : (i + 1) * BS]
        chunk_points = points[i * BS : (i + 1) * BS]

        chunk_data = chunk_data.reshape(
            (chunk_data.shape[0], chunk_data.shape[1], 16 * args.KNNstep, 3)
        )
        chunk_data = chunk_data[:, :, [i * args.KNNstep for i in range(16)]]
        prediction = model(
            chunk_points.double().to(device), chunk_data.double().to(device)
        ).flatten(start_dim=0, end_dim=1)

        if i == 0:
            prediction_cat = prediction.cpu().detach().numpy()
        else:
            prediction_cat = np.r_[
                "0", prediction_cat, prediction.cpu().detach().numpy()
            ]

        batch_end_time = time.time()
        print(f"Batch {i+1} time: {batch_end_time - batch_start_time}")
        total_time += batch_end_time - batch_start_time

    print(f"Total time: {total_time}")

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
