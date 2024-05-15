import torch
import torch.nn as nn
from modules.blocks.model import (
    UNet,
    get_timestep_embedding,
    nonlinearity,
    Downsample,
    Upsample,
    ResnetBlock,
)


class ConditionedUNet(UNet):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=...,
        num_res_blocks,
        attn_resolutions,
        dropout=0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep: bool = True,
        use_linear_attn: bool = False,
        attn_type="vanilla",
    ):
        super().__init__(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=in_channels,
            resolution=resolution,
            use_timestep=use_timestep,
            use_linear_attn=use_linear_attn,
            attn_type=attn_type,
        )

    def forward(self, x, t=None, context=None):
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def skip_cond(self, skip_inputs, skip_conds):
        """
        This block was inspired by 'SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing', see Figure 4
        """
        assert skip_inputs.shape == skip_conds.shape


class MergeBlock(UNet):
    """
    将条件融合进输入的UNet基类
    """

    def __init__(
        self,
        *,
        in_channels,
        out_ch,
        ch_mult=(1, 2, 4),
        emb_ch,
        resamp_with_conv=True,
        dropout=0.0,
    ):
        super().__init__()
        assert in_channels == out_ch
        self.emb_ch = emb_ch
        self.in_channels = in_channels
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.temb_ch = emb_ch * 4

        self.conv_inputs = [nn.Conv2d(in_channels, in_channels, 3, padding=1)]
        self.conv_conds = [nn.Conv2d(in_channels, in_channels, 3, padding=1)]
        in_ch_mult = (1,) + tuple(ch_mult)
        for last_id, now_id in zip(in_ch_mult, ch_mult):
            self.conv_inputs.append(
                nn.ModuleList(
                    [
                        Downsample(emb_ch * last_id, resamp_with_conv),
                        nn.Conv2d(emb_ch * last_id, emb_ch * now_id, 3, padding=1),
                    ]
                )
            )
            self.conv_conds.append(
                nn.ModuleList(
                    [
                        Downsample(emb_ch * last_id, resamp_with_conv),
                        nn.Conv2d(emb_ch * last_id, emb_ch * now_id, 3, padding=1),
                    ]
                )
            )
        self.mid = ResnetBlock(
            emb_ch * ch_mult[-1],
            emb_ch * ch_mult[-1],
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.init_conv_out(emb_ch, ch_mult, out_ch, resamp_with_conv)

    def init_conv_out(self, emb_ch, ch_mult, out_ch, resamp_with_conv):
        self.conv_out = []
        out_ch_mult = (ch_mult[-1],) + ch_mult[::-1]
        for last_id, now_id in zip(out_ch_mult, ch_mult[::-1]):
            self.conv_out.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(emb_ch * last_id, emb_ch * now_id, 3, padding=1),
                        Upsample(emb_ch * last_id, resamp_with_conv),
                    ]
                )
            )

        self.conv_out.append([nn.Conv2d(out_ch, out_ch, 3, padding=1)])

    def forward(self, inputs, conds):
        pass
