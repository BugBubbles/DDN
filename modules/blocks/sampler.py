import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import random, math


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleSquare(nn.Module):
    def __init__(
        self,
        use_mlp=False,
        init_gain=0.02,
        nc=256,
        patch_w=4,
    ):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleSquare, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_gain = init_gain
        self.patch_w = patch_w
        self.patch_size = self.patch_w * self.patch_w

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1] * self.patch_size
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)]
            )
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init.normal_(self, 0.0, self.init_gain)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W, C = feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[1]
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) #(B, H*W, C)
            feat_reshape = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
            sample = torch.zeros(
                (B, num_patches, self.patch_w * self.patch_w * C),
                device=feats[0].device,
            )
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.zeros((num_patches, 2), device=feat[0].device)
                    for i in range(0, num_patches):
                        patch_id[i, 0] = random.randint(
                            0, feat_reshape.shape[1] - self.patch_w
                        )
                        patch_id[i, 1] = random.randint(
                            0, feat_reshape.shape[2] - self.patch_w
                        )
                    # patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                # patch_id shape: 256
                # x_sample shape: B, num, patchsize, C
                for num, id in enumerate(patch_id):
                    patch = feat_reshape[
                        :,
                        int(id[0]) : int(id[0]) + self.patch_w,
                        int(id[1]) : int(id[1]) + self.patch_w,
                        :,
                    ]
                    patch = patch.contiguous().view(B, -1)
                    sample[:, num, :] = patch  # reshape(-1, x.shape[1])
                # final x_sample: B*num, patchsize*C
                x_sample = sample.flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            return_feats.append(x_sample)
        return return_feats, return_ids


class PatchSampleNonlocalOneGroup(nn.Module):
    def __init__(
        self,
        use_mlp=False,
        init_gain=0.02,
        nc=256,
    ):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleNonlocalOneGroup, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_gain = init_gain
        self.patch_size = 16
        self.half = 8
        self.search_size = 40
        self.stride = 1

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                *[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)]
            )
            setattr(self, "mlp_%d" % mlp_id, mlp)
        init.normal_(self, 0.0, self.init_gain)
        self.mlp_init = True

    # patch_ids here is different with the former ones
    # it means the (x,y) loc in 256x256 raw images
    def forward(self, feats, num_patches=256, patch_ids=None):
        return_feats = []
        if patch_ids is None:  # calculate the num_patches non-local keys in raw image
            # 1. sample one loc in the img
            # 2. slide window to get patches and patch_ids
            # 3. calculate distances
            # 4. top k and gather locs

            # Nonlocal in tensor
            img = feats[0]
            B, H, W = img.shape[0], img.shape[2], img.shape[3]
            sample_loc = [
                random.randint(self.patch_size, H - self.patch_size),
                random.randint(self.patch_size, W - self.patch_size),
            ]
            sample_patch = getTensorPatch(img, sample_loc, self.half)
            locs, patches = getTensorLocs(
                img,
                sample_loc,
                self.patch_size,
                self.search_size,
                self.stride,
                img.device,
            )
            diff_pow = torch.pow((patches - sample_patch), 2)
            diff_sum = torch.sum(torch.sum(torch.sum(diff_pow, 2), 2), 1)
            diff_sqr = torch.sqrt(diff_sum)  # length * 1
            score, index = torch.topk(diff_sqr, num_patches, largest=False, sorted=True)

            patch_ids = locs[index]
            patch_ids = patch_ids.to(torch.long)
        # debug_seeNonlocal(feats[0],patch_ids,self.half)
        # (num_patches, 2)
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat_patch_loc = torch.floor_divide((patch_ids * H), 256)
            patch_id = feat_patch_loc[:, 0] * W + feat_patch_loc[:, 1]  # 256
            # patch_id shape: num_patches
            patch_id = patch_id.to(torch.long)
            x_sample = feat_reshape[:, patch_id, :].flatten(
                0, 1
            )  # reshape(-1, x.shape[1])

            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = mlp(x_sample)
            # return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            return_feats.append(x_sample)
        return_ids = patch_ids
        return return_feats, return_ids


def getNumpyPatch(ImgNumpy, loc, half):
    return ImgNumpy[loc[0] - half : loc[0] + half, loc[1] - half : loc[1] + half, :]


def getTensorLocs(tensor, center_patch_loc, patch_size, search_size, stride, device):
    height, width = tensor.shape[2], tensor.shape[3]
    half = int(patch_size / 2)
    search_half = int(search_size / 2)
    starting_loc_h = (
        half
        if (center_patch_loc[0] - search_half - half < 0)
        else center_patch_loc[0] - search_half
    )
    starting_loc_w = (
        half
        if (center_patch_loc[1] - search_half - half < 0)
        else center_patch_loc[1] - search_half
    )
    ending_loc_h = (
        (height - half)
        if (center_patch_loc[0] + search_half + half > height)
        else center_patch_loc[0] + search_half
    )
    ending_loc_w = (
        (width - half)
        if (center_patch_loc[1] + search_half + half > width)
        else center_patch_loc[1] + search_half
    )
    patches = torch.zeros(
        (tensor.shape[0], tensor.shape[1], patch_size, patch_size), device=device
    )
    locs = torch.zeros((1, 2), device=device)
    count = 0
    for h in range(starting_loc_h, ending_loc_h, stride):
        for w in range(starting_loc_w, ending_loc_w, stride):
            locs = torch.cat((locs, torch.tensor([[h, w]], device=device)), 0)
            patches = torch.cat((patches, getTensorPatch(tensor, [h, w], half)), 0)
            count += 1
    return locs[1:], patches[1:]


def getTensorPatch(Tensor, loc, half):
    return Tensor[:, :, loc[0] - half : loc[0] + half, loc[1] - half : loc[1] + half]
