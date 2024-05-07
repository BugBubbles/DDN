from modules.blocks.model import UNet
import pytorch_lightning as pl
import torch


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
    def __init__(self, *, ignore_keys, ckpt_path, monitor, ddconfig, scheduler_config):
        self.model = CoreUNet(**ddconfig)
        self.use_scheduler = scheduler_config is not None
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

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, "image")
        y = self.get_input(batch, "target")
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
