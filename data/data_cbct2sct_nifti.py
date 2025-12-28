# data_cbct2sct_nifti.py
import os
from os.path import join
from typing import Tuple, List

import numpy as np
import torch
# Ensure the import path below is correct according to your project structure
from data.dataset_cbct2sct_nifti import CBCT2SCTNiftiDataset


def split_ids(root: str, val_ratio: float = 0.2, seed: int = 123) -> Tuple[List[str], List[str]]:
    """Filter only folders containing both cbct.nii.gz and ct.nii.gz, then split into train/val sets."""
    all_dirs = [d for d in os.listdir(root) if os.path.isdir(join(root, d))]
    case_ids = []
    for d in sorted(all_dirs):
        case_dir = join(root, d)
        cbct_ok = os.path.exists(join(case_dir, "cbct.nii.gz"))
        ct_ok = os.path.exists(join(case_dir, "ct.nii.gz"))
        if cbct_ok and ct_ok:
            case_ids.append(d)

    if not case_ids:
        raise RuntimeError(f"No valid cases found in {root} (must contain both cbct.nii.gz and ct.nii.gz)")

    print(f"[split_ids] Found {len(case_ids)} valid cases: {case_ids}")

    rng = np.random.RandomState(seed)
    rng.shuffle(case_ids)
    n_val = max(1, int(len(case_ids) * val_ratio))
    return case_ids[n_val:], case_ids[:n_val]


# --- Core Modification: Changed parameter from plural windows to singular window, type from List[Tuple] to Tuple ---
def get_training_set_nifti(data_dir: str,
                           cbct_window: Tuple[float, float],
                           ct_window: Tuple[float, float],
                           nFrames: int = 5,
                           patch_size: int = 192,
                           augment: bool = True,
                           verbose: bool = True,
                           min_hw: int = 192,
                           val_ratio: float = 0.2,
                           seed: int = 123
                           ) -> CBCT2SCTNiftiDataset:
    """Construct training dataset (Dataset object)."""
    train_ids, _ = split_ids(data_dir, val_ratio=val_ratio, seed=seed)
    return CBCT2SCTNiftiDataset(
        root_dir=data_dir,
        split_ids=train_ids,
        nFrames=nFrames,
        augment=augment,
        patch_size=patch_size,
        cbct_window=cbct_window, # <--- Core Modification
        ct_window=ct_window,   # <--- Core Modification
        min_hw=min_hw,
        verbose=verbose,
    )


# --- Core Modification: Changed parameter from plural windows to singular window, type from List[Tuple] to Tuple ---
def get_eval_set_nifti(data_dir: str,
                       cbct_window: Tuple[float, float],
                       ct_window: Tuple[float, float],
                       nFrames: int = 5,
                       verbose: bool = True,
                       min_hw: int = 0,
                       patch_size: int = 0,
                       val_ratio: float = 0.2,
                       seed: int = 123
                       ) -> CBCT2SCTNiftiDataset:
    """Construct evaluation/validation dataset (Dataset object)."""
    _, val_ids = split_ids(data_dir, val_ratio=val_ratio, seed=seed)
    return CBCT2SCTNiftiDataset(
        root_dir=data_dir,
        split_ids=val_ids,
        nFrames=nFrames,
        augment=False,
        patch_size=patch_size,
        cbct_window=cbct_window, # <--- Core Modification
        ct_window=ct_window,   # <--- Core Modification
        min_hw=min_hw,
        verbose=verbose,
    )


# =============================================================================
# == Test Entry (This file can be run directly for testing)
# =============================================================================
if __name__ == "__main__":
    import argparse

    # --- Core Modification: Changed from multiple window width list to single window width tuple ---
    DEFAULT_CBCT_WINDOW = (-1500, 2500)
    DEFAULT_CT_WINDOW = (-1000, 1500)

    parser = argparse.ArgumentParser("data_cbct2sct_nifti (Single Window Width Version) Test")
    parser.add_argument("--root_dir", type=str, default=r" ")  # Please modify to your data path
    parser.add_argument("--nframes", type=int, default=5)
    parser.add_argument("--patch", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print("=== Construct Training Set (Single Window Width) ===")
    # --- Core Modification: Updated function call parameters ---
    train_set = get_training_set_nifti(
        data_dir=args.root_dir,
        cbct_window=DEFAULT_CBCT_WINDOW,
        ct_window=DEFAULT_CT_WINDOW,
        nFrames=args.nframes,
        patch_size=args.patch,
        augment=True,
        verbose=True,
    )
    print("Training set size (total slice samples):", len(train_set))

    print("\n=== Construct Validation Set (Single Window Width) ===")
    val_set = get_eval_set_nifti(
        data_dir=args.root_dir,
        cbct_window=DEFAULT_CBCT_WINDOW,
        ct_window=DEFAULT_CT_WINDOW,
        nFrames=args.nframes,
        verbose=True,
    )
    print("Validation set size (total slice samples):", len(val_set))

    print("\n--- DataLoader Connectivity Test ---")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))
    gt_b, inp_b, nbs_b, bic_b = batch

    # --- Core Modification: Updated print information, channel number should be 1 ---
    print(f"[Train] Input (CBCT) shape: {inp_b.shape}  <-- Single window width, channel number should be 1")
    print(f"[Train] Ground Truth (CT) shape:   {gt_b.shape}  <-- Single window width, channel number should be 1")
    print(f"[Train] Neighbors (list length): {len(nbs_b)}, each neighbor shape: {nbs_b[0].shape}")

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    vbatch = next(iter(val_loader))
    v_gt, v_inp, v_nbs, v_bic = vbatch
    print(f"[Val]   Input (CBCT) shape: {v_inp.shape}")
    print(f"[Val]   Ground Truth (CT) shape:   {v_gt.shape}")