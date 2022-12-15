import torch
from torch import nn
from point_transformer_pytorch import (
    PointTransformerLayer,
    MultiheadPointTransformerLayer,
)
from point_transformer_pytorch.point_transformer_pytorch import batched_index_select


def downsample(point, num_points, device):
    indices = torch.randperm(point.shape[1])[:num_points]
    downsampled_points = torch.cat(
        (
            point[:, indices],
            torch.zeros((point.shape[0], point.shape[1] - num_points, 3)).to(device),
        ),
        1,
    )
    rel_pos = (downsampled_points[:, :, None, :] - point[:, None, :, :])[:, :128]
    rel_dist = rel_pos.norm(dim=-1)
    dist, indices = rel_dist.topk(16, largest=False)
    return (downsampled_points[:, :128], indices, dist)


def upsample(downsampled_points, upsampled_points, device):
    downsampled_points = torch.cat(
        (
            downsampled_points,
            torch.zeros(
                (
                    downsampled_points.shape[0],
                    upsampled_points.shape[1] - downsampled_points.shape[1],
                    3,
                )
            ).to(device),
        ),
        1,
    )
    rel_pos = (upsampled_points[:, :, None, :] - downsampled_points[:, None, :, :])[
        :, :, : downsampled_points.shape[1]
    ]
    rel_dist = rel_pos.norm(dim=-1)
    dist, indices = rel_dist.topk(16, largest=False)
    return (indices, dist)


class model0_0(nn.Module):
    # Model with transformer interpolation in upsampling
    def __init__(self, *, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.attn1 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 1),
        )

    def forward(self, original_points, data):
        data = data.flatten(start_dim=0, end_dim=1)
        # Self-attention of input points wrt target points
        sa1 = self.attn1(data, data[:, :, 0:3])
        sa1 = torch.stack(
            sa1.flatten(start_dim=-2, end_dim=-1).split(512, dim=0), dim=0
        )
        output = self.mlp(sa1)

        return output


class model0_1(nn.Module):
    # Model with transformer interpolation in upsampling
    def __init__(self, *, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.attn1 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn2 = PointTransformerLayer(
            dim=16 * 3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16,
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 3, 16 * 3), nn.ReLU(), nn.Linear(16 * 3, 16 * 3), nn.ReLU()
        )
        self.mlp_output = nn.Sequential(
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 1),
        )

    def forward(self, original_points, data):
        data = data.flatten(start_dim=0, end_dim=1)
        # Self-attention of input points wrt target points
        sa1 = self.attn1(data, data[:, :, 0:3])
        sa1 = torch.stack(
            sa1.flatten(start_dim=-2, end_dim=-1).split(512, dim=0), dim=0
        )
        sa1 = self.mlp(sa1)

        # Self-attention within target points
        sa2 = self.attn2(sa1, original_points)

        output = self.mlp_output(sa2)

        return output


class model1(nn.Module):
    # Model with transformer interpolation in upsampling
    def __init__(self, *, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.attn1 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn2 = PointTransformerLayer(
            dim=16 * 3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16,
        )
        self.attn3 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn4 = PointTransformerLayer(
            dim=16 * 3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn5 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 3, 16 * 3), nn.ReLU(), nn.Linear(16 * 3, 16 * 3), nn.ReLU()
        )
        self.mlp_output = nn.Sequential(
            nn.Linear(16 * 6, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 1),
        )
        self.maxpool = nn.MaxPool2d((1, 16), stride=(1, 16))

    def forward(self, original_points, data):
        data = data.flatten(start_dim=0, end_dim=1)
        # Self-attention of input points wrt target points
        sa1 = self.attn1(data, data[:, :, 0:3])
        sa1 = torch.stack(
            sa1.flatten(start_dim=-2, end_dim=-1).split(512, dim=0), dim=0
        )
        sa1 = self.mlp(sa1)

        # Self-attention within target points
        sa2 = self.attn2(sa1, original_points)
        sa2 = self.mlp(sa2)

        # Downsampling
        pivot_points, indices, dist = downsample(original_points, 128, self.device)
        downsampled_sa2 = batched_index_select(sa2, indices).flatten(
            start_dim=0, end_dim=1
        )
        neighbour_points = batched_index_select(original_points, indices).flatten(
            start_dim=0, end_dim=1
        )

        sa3 = self.attn3(
            self.maxpool(downsampled_sa2),
            neighbour_points,
        )
        sa3 = torch.stack(
            sa3.flatten(start_dim=-2, end_dim=-1).split(128, dim=0), dim=0
        )
        sa4 = self.attn4(
            sa3,
            pivot_points,
        )

        # Upsampling
        indices, dist = upsample(pivot_points, original_points, self.device)
        upsampled_sa4 = batched_index_select(sa4, indices).flatten(
            start_dim=0, end_dim=1
        )
        upsampled_points = batched_index_select(pivot_points, indices).flatten(
            start_dim=0, end_dim=1
        )
        upsampled_sa4 = self.attn5(
            self.maxpool(upsampled_sa4),
            upsampled_points,
        )
        upsampled_sa4 = torch.stack(
            upsampled_sa4.flatten(start_dim=-2, end_dim=-1).split(512, dim=0), dim=0
        )
        upsampled_sa4 = torch.cat((upsampled_sa4, sa2), axis=-1)

        output = self.mlp_output(upsampled_sa4)

        return output


class model2(nn.Module):
    # Model with trilinear interpolation for upsampling
    def __init__(self, *, batch_size, device):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

        self.attn1 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn2 = PointTransformerLayer(
            dim=16 * 3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
            num_neighbors=16,
        )
        self.attn3 = PointTransformerLayer(
            dim=3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.attn4 = PointTransformerLayer(
            dim=16 * 3,
            pos_mlp_hidden_dim=64,
            attn_mlp_hidden_mult=4,
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 3, 16 * 3), nn.ReLU(), nn.Linear(16 * 3, 16 * 3), nn.ReLU()
        )
        self.mlp_output = nn.Sequential(
            nn.Linear(16 * 6, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 16 * 3),
            nn.ReLU(),
            nn.Linear(16 * 3, 1),
        )
        self.maxpool = nn.MaxPool2d((1, 16), stride=(1, 16))

    def forward(self, original_points, data):
        data = data.flatten(start_dim=0, end_dim=1)
        # Self-attention of input points wrt target points
        sa1 = self.attn1(data, data[:, :, 0:3])
        sa1 = torch.stack(
            sa1.flatten(start_dim=-2, end_dim=-1).split(512, dim=0), dim=0
        )
        sa1 = self.mlp(sa1)

        # Self-attention within target points
        sa2 = self.attn2(sa1, original_points)
        sa2 = self.mlp(sa2)

        # Downsampling
        pivot_points, indices, _ = downsample(original_points, 128, self.device)
        downsampled_sa2 = batched_index_select(sa2, indices).flatten(
            start_dim=0, end_dim=1
        )
        neighbour_points = batched_index_select(original_points, indices).flatten(
            start_dim=0, end_dim=1
        )

        sa3 = self.attn3(
            self.maxpool(downsampled_sa2),
            neighbour_points,
        )
        sa3 = torch.stack(
            sa3.flatten(start_dim=-2, end_dim=-1).split(128, dim=0), dim=0
        )
        sa4 = self.attn4(
            sa3,
            pivot_points,
        )

        # Upsampling
        indices, dist = upsample(pivot_points, original_points, self.device)
        upsampled_sa4 = batched_index_select(sa4, indices)
        # trilinear interpolation
        dist_reverse = (
            torch.max(dist, -1)[0].unsqueeze(-1).repeat([1, 1, 16]) - dist + 0.0001
        )
        dist_norm = dist_reverse / torch.sum(dist_reverse, -1).unsqueeze(-1).repeat(
            [1, 1, 16]
        )
        dist_norm = dist_norm.unsqueeze(-1).repeat([1, 1, 1, 16 * 3])
        upsampled_sa4 *= dist_norm
        upsampled_sa4 = self.maxpool(upsampled_sa4)
        upsampled_sa4 = upsampled_sa4.flatten(start_dim=-2, end_dim=-1)
        upsampled_sa4 = torch.cat((upsampled_sa4, sa2), axis=-1)

        output = self.mlp_output(upsampled_sa4)

        return output


# Global variables in dictionaries for the models used
model_dict = {
    "model0_0": model0_0,
    "model0_1": model0_1,
    "model1": model1,
    "model2": model2,
}
