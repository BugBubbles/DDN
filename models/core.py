from modules.blocks.model import UNet
from modules.patcher.sampler import (
    PatchSampleLocalOneGroup,
    PatchSampleNonlocalOneGroup,
)
from modules.blocks.cl import Discriminator, Generator
from modules.losses.nce import PatchNCELoss, DisNCELoss
from modules.distributions import normal_kl, DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import instantiate_from_config


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


class Restorer(pl.LightningModule):
    def __init__(
        self,
        *,
        lambdas,
        input_key="image",
        ignore_keys=[],
        ckpt_path,
        monitor,
        first_stage_config,
        ddconfig,
        unet_config,
        scheduler_config,
        lossconfig,
        sampler_config,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.core_unet = instantiate_from_config(unet_config)
        # self.local_sampler = instantiate_from_config(sampler_config)
        self.nonlocal_sampler = instantiate_from_config(sampler_config)
        self.use_scheduler = scheduler_config is not None
        self.loss_consi_ = instantiate_from_config(lossconfig)
        self.loss_loc_ = []
        self.loss_layer_ = DisNCELoss()
        for _ in lossconfig.pop("gen_nce_layers"):
            self.loss_loc_.append(PatchNCELoss())
        self.mom_gen = Generator(**ddconfig)
        self.mom_dis = Discriminator(**ddconfig)
        self.sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.lambdas = lambdas
        self.input_key = input_key
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        # load the pretrained Auto-Encoder model
        self.instantiate_first_stage(first_stage_config)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train(False)
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(
                f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
            )
        return z

    def get_input(self, batch, key_):
        x = batch[key_]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def forward(self, x):
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        for _ in range(6):
            z = self.core_unet(z)
        noise = self.decode_first_stage(z)
        x_hat = noise + x
        return x_hat, noise

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.input_key)
        x_hat, noise = self(x)
        # opt_core, opt_dis, opt_gen = self.optimizers()

        # automatic backward without using closure
        opt, opt_de = self.optimizers()

        def unet_closure():
            loss, loss_dict = self.loss(x_hat, x, noise, *self.lambdas, stage="train")
            self.log_dict(
                loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True
            )
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()

        def de_closure():
            loss, loss_dict = self.loss(x_hat, x, noise, *self.lambdas, stage="disc")
            # self.log_dict(
            #     loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True
            # )
            # self.log(
            #     "discloss",
            #     loss,
            #     prog_bar=False,
            #     logger=True,
            #     on_step=False,
            #     on_epoch=True,
            # )
            opt_de.zero_grad()
            self.manual_backward(loss)
            opt_de.step()

        opt.step(closure=unet_closure)
        if batch_idx % 4 == 0:
            opt_de.step(closure=de_closure)

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.input_key)
        x_hat, noise = self(x)
        _, loss_dict = self.loss(x_hat, x, noise, *self.lambdas, stage="validate")
        self.log_dict(
            loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True
        )
        return self.log_dict

    def loss(
        self,
        x_hat,
        x,
        noise,
        lambda_loc,
        lambda_layer,
        lambda_consis,
        lambda_destrip,
        lambda_penalty,
        *,
        stage="train",
    ):
        loss_loc = self.calculate_loc_loss(x_hat, noise, x)
        loss_layer = self.calculate_layer_loss(x_hat, noise, x)
        loss_consis = self.loss_consi_(x_hat, x)
        loss_destrip, _, _ = self.calculate_strip_loss(x_hat, noise, x, 0.5)
        loss_penalty = F.l1_loss(noise, torch.zeros_like(noise))
        loss = (
            lambda_consis * loss_consis.mean()
            + lambda_loc * loss_loc
            + lambda_layer * loss_layer
            + lambda_destrip * loss_destrip
            + lambda_penalty * loss_penalty
        )
        loss_dict = {
            f"{stage}/loss": loss,
            f"{stage}/loss_loc": loss_loc,
            f"{stage}/loss_layer": loss_layer,
            f"{stage}/loss_consis": loss_consis.mean(),
            f"{stage}/loss_destrip": loss_destrip,
            f"{stage}/loss_penalty": loss_penalty,
        }
        return loss, loss_dict

    def calculate_strip_loss(self, x_hat, noise, x, lambda_):
        chs = x_hat.shape[1]
        sobel_x = self.sobel_x.repeat(chs, chs, 1, 1).to(x.dtype).to(device=x.device)
        sobel_y = self.sobel_y.repeat(chs, chs, 1, 1).to(x.dtype).to(device=x.device)
        x_hat_grad_x = F.conv2d(x_hat, sobel_x, padding=1)
        noise_grad_y = F.conv2d(noise, sobel_y, padding=1)
        loss = (1 - lambda_) * F.l1_loss(
            x_hat_grad_x, torch.zeros_like(x_hat_grad_x)
        ) + lambda_ * F.l1_loss(noise_grad_y, torch.zeros_like(noise_grad_y))
        return loss, x_hat_grad_x, noise_grad_y

    def calculate_layer_loss(self, x_hat, noise, x, adv_nce_layers=[0, 3, 7, 11]):
        # 这个在实际使用中效果是反向的，应该用得不对
        feat_x_hat = self.mom_dis(x_hat, adv_nce_layers, encode_only=True)
        feat_noise = self.mom_dis(noise, adv_nce_layers, encode_only=True)

        feat_B_pool, _ = self.nonlocal_sampler(feat_x_hat, "layer_B", 8, None)
        feat_N_pool, _ = self.nonlocal_sampler(feat_noise, "layer_N", 128, None)

        total_dis_loss = 0.0
        for f_b, f_r, nce_layer in zip(feat_B_pool, feat_N_pool, adv_nce_layers):
            loss = self.loss_layer_(f_b, f_r)
            total_dis_loss += loss.mean()

        return total_dis_loss / len(adv_nce_layers)

    def calculate_loc_loss(self, x_hat, noise, x, gen_nce_layers=[0, 2, 4, 8, 12]):
        # 对去噪得到的图像再作一次卷积提取特征图，我感觉有点喧宾夺主，反而变成训练这两个不需要有卷积网络去了
        # 事实证明我的想法是错误的，这个卷积网络真的有效。
        feat_x_hat = self.mom_gen(x_hat, gen_nce_layers, encode_only=True)
        feat_x = self.mom_gen(x, gen_nce_layers, encode_only=True)
        # feat_b = [torch.flip(fq, [3]) for fq in feat_x_hat]

        feat_o = feat_x
        feat_o_pool, sample_ids = self.nonlocal_sampler(feat_o, "loc_O", 256, None)
        feat_b_pool, _ = self.nonlocal_sampler(feat_x_hat, "loc_B", 256, sample_ids)

        total_nce_loss = 0.0
        for f_b, f_o, layer, nce_layer in zip(
            feat_b_pool, feat_o_pool, self.loss_loc_, gen_nce_layers
        ):
            loss = layer(f_b, f_o)
            total_nce_loss += loss.mean()

        return total_nce_loss / len(gen_nce_layers)

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(
            self.core_unet.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_de = torch.optim.Adam(
            list(self.mom_dis.parameters()) + list(self.mom_gen.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt, opt_de], scheduler
        return opt, opt_de

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        # TODO fix this function
        log = dict()
        x = self.get_input(batch, self.input_key)
        x = x.to(self.device)
        if not only_inputs:
            x_hat, noise = self(x)
            _, grad_x, grad_y = self.calculate_strip_loss(x_hat, noise, x, 0.5)
            if x.shape[1] > 3:
                # colorize with random projection
                assert x_hat.shape[1] > 3
                x = self.to_rgb(x)
                x_hat = self.to_rgb(x_hat)
                noise = self.to_rgb(noise)
            log["denoised"] = x_hat
            log["noise"] = noise
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.input_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
