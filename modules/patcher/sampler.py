from typing import Iterator
import torch
import torch.nn as nn
from torch.nn import init
import random, math
from typing_extensions import Union, List


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(2, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + 1e-7)
        return out


class Sampler(nn.Module):
    def __init__(self, use_mlp=False, emb_ch=256, patch_w=4):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.emb_ch = emb_ch
        self.patch_w = patch_w
        self.mlp = nn.ModuleDict()

    def create_mlp(self, patch_ids, inputs, id: str):
        self.mlp[f"proj_{id}"] = nn.ModuleList()
        for feat in inputs:
            input_nc = feat.shape[1] * patch_ids.shape[1]
            mlp = nn.Sequential(
                *[
                    nn.Linear(input_nc, self.emb_ch),
                    nn.ReLU(),
                    nn.Linear(self.emb_ch, input_nc),
                ]
            ).to(feat.device)
            self.mlp[f"proj_{id}"].append(mlp)

    def forward(self, inputs, num_patches, patch_ids):
        raise NotImplementedError


class PatchSampleLocalOneGroup(Sampler):
    """
    Patch in the same position in different images
    """

    def __init__(self, use_mlp=False, emb_ch=256, patch_w=4, **kwargs):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__(use_mlp=use_mlp, emb_ch=emb_ch, patch_w=patch_w)
        self.patch_size = self.patch_w * self.patch_w

    def forward(
        self,
        inputs: list[torch.TensorType],
        num_patches=64,
        patch_ids=Union[List, None],
    ):
        return_ids = []
        outputs = []
        if self.use_mlp and f"proj_{id}" not in self.mlp.keys():
            self.create_mlp(inputs)
        for feat_id, feat in enumerate(inputs):
            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            # feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) #(B, H*W, C)
            feat_reshape = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
            sample = torch.zeros(
                (B, num_patches, self.patch_w * self.patch_w * C),
                device=inputs[0].device,
            )
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.zeros((num_patches, 2), device=feat[0].device)
                    for i in range(num_patches):
                        patch_id[i, 0] = random.randint(0, H - self.patch_w)
                        patch_id[i, 1] = random.randint(0, W - self.patch_w)
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
                assert f"proj_{id}" in self.mlp.keys()
                # mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = self.mlp[f"proj_{id}"][feat_id](x_sample.view(B, -1))
                x_sample = x_sample.reshape(B, num_patches, C)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            outputs.append(x_sample)
        return outputs, return_ids


class PatchSampleNonlocalOneGroup(Sampler):
    """
    Patch in different positions in different images
    """

    def __init__(self, use_mlp=False, emb_ch=256, patch_w=4, search_size=40):
        super().__init__(use_mlp=use_mlp, emb_ch=emb_ch, patch_w=patch_w)
        self.half_ = self.patch_w // 2
        self.search_size = search_size
        self.stride = 1

    def forward(self, inputs, id: str, num_patches=256, patch_ids=None):
        outputs = []
        if patch_ids is None:  # calculate the num_patches non-local keys in raw image
            # 1. sample one loc in the img
            # 2. slide window to get patches and patch_ids
            # 3. calculate distances
            # 4. top k and gather locs

            # Nonlocal in tensor
            img = inputs[0]
            B, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
            sample_loc = [
                random.randint(self.patch_w, H - self.patch_w),
                random.randint(self.patch_w, W - self.patch_w),
            ]
            sample_patch = getTensorPatch(img, sample_loc, self.half_)
            locs, patches = getTensorLocs(
                img,
                sample_loc,
                self.patch_w,
                self.search_size,
                self.stride,
                img.device,
            )
            diff_pow = torch.pow((patches - sample_patch), 2)
            diff_sum = torch.sum(torch.sum(torch.sum(diff_pow, 3), 3), 2)
            diff_sqr = torch.sqrt(diff_sum)  # length * 1
            score, index = torch.topk(
                diff_sqr, num_patches, dim=1, largest=False, sorted=True
            )
            index = index.repeat(2, 1, 1).permute(1, 2, 0)
            patch_ids = locs.gather(1, index).to(torch.long)
        # debug_seeNonlocal(inputs[0],patch_ids,self.half_)
        # (num_patches, 2)
        if self.use_mlp and f"proj_{id}" not in self.mlp.keys():
            self.create_mlp(patch_ids, inputs, id)
        for feat_id, feat in enumerate(inputs):
            B, C, H, W = feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # (B, H*W, C)
            feat_patch_loc = torch.floor_divide((patch_ids * H), 256)
            patch_id = feat_patch_loc[..., 0] * W + feat_patch_loc[..., 1]  # 256
            # patch_id shape: num_patches
            patch_id = patch_id.to(torch.long).repeat(C, 1, 1).permute(1, 2, 0)
            x_sample = feat_reshape.gather(1, patch_id)

            if self.use_mlp:
                assert f"proj_{id}" in self.mlp.keys()
                # mlp = getattr(self, "mlp_%d" % feat_id)
                x_sample = self.mlp[f"proj_{id}"][feat_id](x_sample.view(B, -1))
                x_sample = x_sample.reshape(B, num_patches, C)
            # return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )
            outputs.append(x_sample)
        return_ids = patch_ids
        return outputs, return_ids


def getNumpyPatch(ImgNumpy, loc, half_):
    return ImgNumpy[loc[0] - half_ : loc[0] + half_, loc[1] - half_ : loc[1] + half_, :]


def getTensorLocs(inputs, center_patch_loc, patch_w, search_size, stride, device):
    height, width = inputs.shape[2], inputs.shape[3]
    half_ = patch_w // 2
    search_half = search_size // 2
    starting_loc_h = (
        half_
        if (center_patch_loc[0] - search_half - half_ < 0)
        else center_patch_loc[0] - search_half
    )
    starting_loc_w = (
        half_
        if (center_patch_loc[1] - search_half - half_ < 0)
        else center_patch_loc[1] - search_half
    )
    ending_loc_h = (
        (height - half_)
        if (center_patch_loc[0] + search_half + half_ > height)
        else center_patch_loc[0] + search_half
    )
    ending_loc_w = (
        (width - half_)
        if (center_patch_loc[1] + search_half + half_ > width)
        else center_patch_loc[1] + search_half
    )
    patches = torch.zeros(
        (inputs.shape[0], 1, inputs.shape[1], patch_w, patch_w), device=device
    )
    locs = torch.zeros((inputs.shape[0], 1, 2), device=device)
    for h in range(starting_loc_h, ending_loc_h, stride):
        for w in range(starting_loc_w, ending_loc_w, stride):
            locs = torch.cat(
                (
                    locs,
                    torch.tensor([[[h, w]]], device=device).repeat(
                        inputs.shape[0], 1, 1
                    ),
                ),
                1,
            )
            patches = torch.cat((patches, getTensorPatch(inputs, [h, w], half_)), 1)
    return locs[:, 1:], patches[:, 1:]


def getTensorPatch(inputs, loc, half_):
    """
    inputs: B, C, H, W
    outputs: B, N, C, H, W
    """
    return (
        inputs[:, :, loc[0] - half_ : loc[0] + half_, loc[1] - half_ : loc[1] + half_]
        .repeat(1, 1, 1, 1, 1)
        .permute(1, 0, 2, 3, 4)
    )
