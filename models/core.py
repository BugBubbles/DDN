from modules.blocks.model import UNet
import pytorch_lightning as pl
import torch
from modules.losses.disnce import DisNCELoss
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
        self.use_scheduler = scheduler_config is not None
        self.loss = instantiate_from_config(lossconfig)
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
        res = self.core_unet(inputs, 0)
        output = res + inputs
        return output

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        x_hat = self(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
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
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(
                    batch_size=N, return_intermediates=True
                )

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
