import numpy as np
import os
import torch
from ... import logging


def open_cache(npz_path, mmap_mode=None):
    try:
        npz = np.load(npz_path, mmap_mode=mmap_mode)
        return npz
    except Exception as e:
        import shutil
        logging.warn(f"remove corrupted npz file: {os.path.abspath(npz_path)} | error: {e}")
        backup_path = str(npz_path) + '.bak'
        if os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(str(npz_path), backup_path)
        return None


def load_cache_from_disk(npz_path, dtype=None, flip_aug=False, mmap_mode=None):
    npz = open_cache(npz_path, mmap_mode=mmap_mode)
    if npz is None:
        return None, None
    emb = npz["emb"]
    flipped_emb = npz["emb_flipped"] if "emb_flipped" in npz and flip_aug else None
    emb = torch.FloatTensor(emb).to(dtype=dtype)
    flipped_emb = torch.FloatTensor(flipped_emb).to(dtype=dtype) if flipped_emb is not None else None

    if torch.any(torch.isnan(emb)):
        emb = torch.where(torch.isnan(emb), torch.zeros_like(emb), emb)
        logging.warn(f"NaN detected in emb: {npz_path}")
    if flipped_emb is not None and torch.any(torch.isnan(flipped_emb)):
        flipped_emb = torch.where(torch.isnan(flipped_emb), torch.zeros_like(flipped_emb), flipped_emb)
        logging.warn(f"NaN detected in flipped emb: {npz_path}")

    return emb, flipped_emb
