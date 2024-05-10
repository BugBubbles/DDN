from modules.blocks.model import UNet
from modules.patcher.sampler import (
    PatchSampleLocalOneGroup,
    PatchSampleNonlocalOneGroup,
)
from modules.blocks.cl import Discriminator, Generator
from modules.losses.nce import PatchNCELoss, DisNCELoss
import pytorch_lightning as pl
import torch
from utils import instantiate_from_config


class CoreUNet(UNet):
    def __init__(
        self,
        *,
        resolution,
        emb_ch,
        input_ch,
        output_ch,
        num_res_blocks,
        attn_resolutions,
        ch_mult=(1, 2, 4, 4),
        dropout=0,
        resamp_with_conv=True,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
    ):
        super().__init__(
            ch=emb_ch,
            out_ch=output_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            resamp_with_conv=resamp_with_conv,
            in_channels=input_ch,
            resolution=resolution,
            use_timestep=use_timestep,
            use_linear_attn=use_linear_attn,
            attn_type=attn_type,
        )


class Restorer(pl.LightningModule):
    def __init__(
        self,
        *,
        ignore_keys=None,
        ckpt_path,
        monitor,
        ddconfig,
        scheduler_config,
        lossconfig,
        sampler_config,
    ):
        super().__init__()
        self.core_unet = CoreUNet(**ddconfig)
        self.local_sampler = PatchSampleLocalOneGroup()
        self.nonlocal_sampler = PatchSampleNonlocalOneGroup()
        self.use_scheduler = scheduler_config is not None
        self.main_loss = instantiate_from_config(lossconfig)
        self.loss_layer_ = []
        self.loss_loc_ = DisNCELoss()
        for nce_layer in lossconfig["gen_nce_layers"]:
            self.loss_layer_.append(PatchNCELoss())
        self.mom_gen = Generator(ddconfig["input_ch"], ddconfig["output_ch"])
        self.mom_dis = Discriminator(ddconfig["input_ch"])
        if self.use_scheduler:
            self.scheduler_config = scheduler_config
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def forward(self, inputs):
        res = self.core_unet(inputs)
        output = res + inputs
        return output, res

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        x_hat, noise = self(x)

        # for clean image x_hat and noise iamge noise, there are local and nonlocal sample for L_location calculation
        loss_loc = self.calculate_loc_loss(x_hat, noise, x)
        loss_layer = self.calculate_layer_loss(x_hat, noise, x)
        loss_consis = self.main_loss(x_hat, x)
        loss_destrip = 
        self.log("train_loss", loss)
        return loss
    def calculate_strip_loss(self,x_hat,noise,x):
        

    def calculate_loc_loss(self, x_hat, noise, x, adv_nce_layers=[0, 3, 7, 11]):
        feat_x_hat = self.mom_dis(x_hat, adv_nce_layers, encode_only=True)
        feat_noise = self.mom_dis(noise, adv_nce_layers, encode_only=True)

        feat_B_pool, _ = self.nonlocal_sampler(feat_x_hat, 8, None)
        feat_N_pool, _ = self.nonlocal_sampler(feat_noise, 128, None)

        total_dis_loss = 0.0
        for f_b, f_r, nce_layer in zip(feat_B_pool, feat_N_pool, adv_nce_layers):
            loss = self.loss_loc_(f_b, f_r)
            total_dis_loss += loss.mean()

        return total_dis_loss / len(adv_nce_layers)

    def calculate_layer_loss(self, x_hat, noise, x, gen_nce_layers=[0, 2, 4, 8, 12]):
        feat_b = x_hat
        feat_b = [torch.flip(fq, [3]) for fq in feat_b]

        feat_o = x
        feat_o_pool, sample_ids = self.nonlocal_sampler(feat_o, 256, None)
        feat_b_pool, _ = self.nonlocal_sampler(feat_b, 256, sample_ids)

        total_nce_loss = 0.0
        for f_b, f_o, layer, nce_layer in zip(
            feat_b_pool, feat_o_pool, self.loss_layer_, gen_nce_layers
        ):
            loss = layer(f_b, f_o)
            total_nce_loss += loss.mean()

        return total_nce_loss / len(gen_nce_layers)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        x_hat = self(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_core = torch.optim.Adam(
            list(
                self.core_unet.parameters(),
            ),
            lr=lr,
            betas=(0.5, 0.9),
        )
        ope_dis = torch.optim.Adam(self.mom_dis.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_core, ope_dis], []

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
