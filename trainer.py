"""xvdp modified for pytorch_lighting 1.4
pytorch_lighting contains more deprecations than (metaphorical expletive)
(#5321) (#6162) (#11578) (#9754)
"""
import os
import argparse
import datetime
from glob import glob
from copy import deepcopy
import torch
from lightning_model import NuWave
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf as OC
from utils.tblogger import TensorBoardLoggerExpanded


# Other DDPM/Score-based model applied EMA
# In our works, there are no significant difference
# Deprecated Callback.on_epoch_end hook in favour of Callback.on_{train/val/test}_epoch_end (#11578)

class EMACallback(Callback):
    def __init__(self, filepath, alpha=0.999, k=3):
        super().__init__()
        self.alpha = alpha
        self.filepath = filepath
        self.k = 3 #max_save
        self.queue = []
        self.last_parameters = None

    @rank_zero_only
    def _del_model(self, removek):
        if os.path.exists(self.filepath.format(epoch=removek)):
            os.remove(self.filepath.format(epoch=removek))

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module,batch, batch_idx,dataloader_idx):
        if hasattr(self, 'current_parameters'):
            self.last_parameters = self.current_parameters
        else:
            self.last_parameters = deepcopy(pl_module.state_dict())

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,dataloader_idx):
        self.current_parameters = deepcopy(pl_module.state_dict())
        for k, v in self.current_parameters.items():
            self.current_parameters[k].copy_(self.alpha * v +
                                             (1. - self.alpha) *
                                             self.last_parameters[k])
        del self.last_parameters
        return

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self.queue.append(trainer.current_epoch)
        torch.save(self.current_parameters,
                   self.filepath.format(epoch=trainer.current_epoch))
        pl_module.print(
            f'{self.filepath.format(epoch = trainer.current_epoch)} is saved')

        while len(self.queue) > self.k:
            self._del_model(self.queue.pop(0))
        return


def train(args):
    hparams = OC.load('hparameter.yaml')
    now = datetime.datetime.now().strftime('%m_%d_%H')
    hparams.name = f"{hparams.log.name}_{now}"
    os.makedirs(hparams.log.tensorboard_dir, exist_ok=True)
    os.makedirs(hparams.log.checkpoint_dir, exist_ok=True)
    model = NuWave(hparams)
    tblogger = TensorBoardLoggerExpanded(hparams)
    ckpt_path = f'{hparams.log.name}_{now}_{{epoch}}'
    print(hparams.log.checkpoint_dir, ckpt_path)

    # note on pytorch_lightning.__version__ '1.1.6'
    # .. warning:: .. deprecated:: 1.0
    #       Use ``dirpath`` + ``filename`` instead. Will be removed in v1.2
    # changed
    # prefix: A string to put at the beginning of checkpoint filename.
    # Removed deprecated checkpoint argument filepath (#5321)
    # Removed deprecated ModelCheckpoint arguments prefix, mode="auto" (#6162)
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.log.checkpoint_dir,
                                          filename=ckpt_path,
                                          verbose=True,
                                          save_last=True,
                                          save_top_k=3,
                                          monitor='val_loss',
                                          mode='min')

    if args.restart:
        ckpt = torch.load(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}.ckpt'))[-1],
                          map_location='cpu')
        ckpt_sd = ckpt['state_dict']
        sd = model.state_dict()
        for k, v in sd.items():
            if k in ckpt_sd:
                if ckpt_sd[k].shape == v.shape:
                    sd[k].copy_(ckpt_sd[k])
    if args.ema:
        ckpt = torch.load(glob(
            os.path.join(hparams.log.checkpoint_dir,
                         f'*_epoch={args.resume_from}_EMA'))[-1],
                          map_location='cpu')
        print(ckpt.keys())
        sd = model.state_dict()
        for k, v in sd.items():
            if k in ckpt:
                if ckpt[k].shape == v.shape:
                    sd[k].copy_(ckpt[k])
        args.resume_from = None

    # pytorch_lightning.utilities.exceptions.MisconfigurationException:
    # Invalid type provided for checkpoint_callback:
    # Expected bool but received <class 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'>.
    # Pass callback instances to the `callbacks` argument in the Trainer constructor instead.
    # (#9754) Deprecate checkpoint_callback from the Trainer constructor in favour of enable_checkpointing
    trainer = Trainer(
        checkpoint_callback=True,
        gpus=hparams.train.gpus,
        accelerator='ddp' if hparams.train.gpus > 1 else None,
        #plugins='ddp_sharded',
        amp_backend='apex',  #
        amp_level='O2',  #
        #num_sanity_val_steps = -1,
        check_val_every_n_epoch=2,
        gradient_clip_val = 0.5,
        max_epochs=200000,
        logger=tblogger,
        progress_bar_refresh_rate=4,
        callbacks=[
            EMACallback(os.path.join(hparams.log.checkpoint_dir,
                        f'{hparams.name}_epoch={{epoch}}_EMA')),
                        checkpoint_callback
                  ],
        resume_from_checkpoint=None
        if args.resume_from == None or args.restart else sorted(
            glob(
                os.path.join(hparams.log.checkpoint_dir,
                             f'*_epoch={args.resume_from}.ckpt')))[-1])
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume_from', type =int,\
            required = False, help = "Resume Checkpoint epoch number")
    parser.add_argument('-s', '--restart', action = "store_true",\
            required = False, help = "Significant change occured, use this")
    parser.add_argument('-e', '--ema', action = "store_true",\
            required = False, help = "Start from ema checkpoint")
    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    train(args)
