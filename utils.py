import os
import glob
import h5py
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from focal_loss.focal_loss import FocalLoss


class LidarData(Dataset):
    def __init__(self, data_type, step_size=1):
        all_data = []
        all_label = []
        all_point = []
        for h5_name in glob.glob(os.path.join("data", "*_data_tr.h5")):
            f = h5py.File(h5_name)
            indices = np.random.choice(1024, size=512, replace=False)

            point = f["point"][:].astype("float32")[:, indices]

            data = f["data"][:].astype("float32")[:, indices]
            data = data.reshape((data.shape[0], data.shape[1], 128, 4))[:, :, :, 0:3]
            data = data - np.repeat(point[:, :, None, :], 128, axis=-2)
            data = data[:, :, [i * step_size for i in range(16)]]

            label = f["label"][:].astype("float32")[:, indices]

            f.close()
            all_point.append(point)
            all_data.append(data)
            all_label.append(label)

        all_point = np.concatenate(all_point, axis=0)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)

        self.point = all_point
        self.data = all_data
        self.label = all_label

    def __getitem__(self, item):
        point = self.point[item]
        data = self.data[item]
        label = self.label[item]
        return point, data, label

    def __len__(self):
        return self.data.shape[0]


class MSIE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        return self.mse((pred / 1000 + 1) ** -1, (true / 1000 + 1) ** -1)


class Focal_Loss(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.focal = FocalLoss(gamma=3)
        self.threshold = threshold

    def forward(self, pred, true):
        true_class = torch.absolute(true)
        true_class[true_class < self.threshold] = 1
        true_class[true_class >= self.threshold] = 0
        pred[pred < 0] = 0
        pred = pred.flatten()
        true_class = true_class.flatten()

        return self.focal((pred / 1000 + 1) ** -1, true_class.to(torch.int64))


class combined_Loss(nn.Module):
    def __init__(self, threshold, weight):
        super().__init__()
        self.msie = MSIE_Loss()
        self.focal = Focal_Loss(threshold)
        self.weight = weight

    def forward(self, pred, true):
        return self.msie(pred, true) + self.weight * self.focal(pred, true)
