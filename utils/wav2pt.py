import multiprocessing as mp
import os
from glob import glob
import torch
import librosa as rosa
from omegaconf import OmegaConf as OC
from tqdm import tqdm

# pylint: disable=no-member
def wav2pt(wav):
    y,_ = rosa.load(wav, sr=hparams.audio.sr, mono=True)
    y,_ = rosa.effects.trim(y, top_db=15)
    pt_name = os.path.splitext(wav)[0]+'.pt'
    pt = torch.tensor(y)
    torch.save(pt, pt_name)
    del y, pt
    return

if __name__=='__main__':
    hparams = OC.load('hparameter.yaml')
    wavs = glob(os.path.join(hparams.data.dir, '*/*.flac'))
    pool = mp.Pool(processes = hparams.train.num_workers)
    with tqdm(total = len(wavs)) as pbar:
        for _ in tqdm(pool.imap_unordered(wav2pt, wavs)):
            pbar.update()

"""
NOTE: mp error after conversion
Traceback (most recent call last):
  File "/home/z/miniconda3/envs/abj/lib/python3.9/multiprocessing/pool.py", line 268, in __del__
  File "/home/z/miniconda3/envs/abj/lib/python3.9/multiprocessing/queues.py", line 372, in put
AttributeError: 'NoneType' object has no attribute 'dumps'
"""