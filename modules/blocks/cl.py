import torch.nn as nn
from modules.blocks.model import ResnetBlock, Downsample, Encoder


class Discriminator(nn.Module):
    def __init__(
        self,
        input_ch,
        emb_ch=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        no_antialias=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_ch (int)  -- the number of channels in input images
            emb_ch (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if no_antialias:
            sequence = [
                nn.Conv2d(input_ch, emb_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True),
            ]
        else:
            sequence = [
                nn.Conv2d(input_ch, emb_ch, kernel_size=4, stride=1, padding=1),
                nn.LeakyReLU(0.2, True),
                Downsample(emb_ch),
            ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(
                        emb_ch * nf_mult_prev,
                        emb_ch * nf_mult,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    norm_layer(emb_ch * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        emb_ch * nf_mult_prev,
                        emb_ch * nf_mult,
                        kernel_size=4,
                        stride=1,
                        padding=1,
                    ),
                    norm_layer(emb_ch * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(emb_ch * nf_mult),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                emb_ch * nf_mult_prev,
                emb_ch * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
            norm_layer(emb_ch * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(emb_ch * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = inputs
            feats = []
            feats.append(inputs)
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    #                     print('encoder only return features',feats[2].shape)
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        """Standard forward."""
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, input_ch, output_ch, emb_ch=32, num_res_blocks=4):
        super(Generator, self).__init__()
        convs = []
        convs.append(nn.Conv2d(input_ch, emb_ch, ksize=3, padding=1, stride=1))
        for _ in range(num_res_blocks):
            convs.append(ResnetBlock(emb_ch))
        convs.append(ConvMid(emb_ch, emb_ch))
        convs.append(nn.Conv2d(emb_ch, output_ch, ksize=3, padding=1, stride=1))
        self.gen = nn.Sequential(*convs)

    def forward(self, inputs, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = inputs
            feats = []
            for layer_id, layer in enumerate(self.gen):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                else:
                    pass
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            out = self.gen(inputs)
            return out


class ConvMid(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(ConvMid, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                input_ch,
                output_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.InstanceNorm2d(output_ch),
            nn.ReLU(),
            nn.Conv2d(
                output_ch,
                output_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
        )

    def forward(self, x):
        return self.conv(x)