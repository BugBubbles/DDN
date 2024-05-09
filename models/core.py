from modules.blocks.model import UNet
from modules.patcher.sampler import (
    PatchSampleLocalOneGroup,
    PatchSampleNonlocalOneGroup,
)
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
    ):
        super().__init__()
        self.core_unet = CoreUNet(**ddconfig)
        self.local_sampler = PatchSampleLocalOneGroup()
        self.nonlocal_sampler = PatchSampleNonlocalOneGroup()
        self.use_scheduler = scheduler_config is not None
        self.loss_ = instantiate_from_config(lossconfig)
        self.loss_loc_ = PatchNCELoss()
        self.loss_layer_ = DisNCELoss()
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
        l_samples, l_ids = self.local_sampler([x_hat, noise])
        nl_samples, nl_ids = self.nonlocal_sampler([x_hat, noise])
        loss_layer = self.loss_layer_(*l_samples) + self.patch_nce_loss(
            *nl_samples[::-1]
        )

        loss_consis = self.loss_(x_hat, x)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        x_hat = self(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.Adam(
            list(self.core_unet.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return opt

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
