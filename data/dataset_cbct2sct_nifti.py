# dataset_cbct2sct_nifti.py
import os
from os.path import join
from typing import List, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data


# ---------------- Utility Functions ----------------
def load_nii(path: str) -> np.ndarray:
    """Load NIfTI file and return float32 array with shape (Z, H, W)."""
    vol = nib.load(path).get_fdata(dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"NIfTI file is not 3D volume data: {path}, shape={vol.shape}")
    return np.transpose(vol, (2, 0, 1))


# --- Core Modification: Function renamed from normalize_hu_multi_window to normalize_hu_single_window ---
def normalize_hu_single_window(v: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    """Apply single window width setting to HU values and return single-channel array."""
    lo, hi = window
    w_v = np.clip(v, lo, hi)
    if hi > lo:
        w_v = (w_v - lo) / (hi - lo)
    else:
        w_v = np.zeros_like(v)
    # Return shape [1, Z, H, W] to maintain dimension consistency
    return np.expand_dims(w_v, axis=0).astype(np.float32)


def clamp_index(z: int, zmin: int, zmax: int) -> int:
    """Clamp index to [zmin, zmax] (boundary reuse)."""
    return min(max(z, zmin), zmax)


# ---------------- Dataset Class ----------------
class CBCT2SCTNiftiDataset(data.Dataset):
    """
    Single window width version:
    - Both CBCT and CT are normalized to single-channel tensors.
    - Returned gt, input, and neighbors all have single-channel shapes, e.g., [1, H, W].
    """

    # --- Core Modification: __init__ parameters changed from plural windows to singular window ---
    def __init__(self,
                 root_dir: str,
                 split_ids: List[str],
                 cbct_window: Tuple[float, float],
                 ct_window: Tuple[float, float],
                 nFrames: int = 5,
                 augment: bool = False,
                 patch_size: int = 0,
                 min_hw: int = 0,
                 verbose: bool = True,
                 ):
        super().__init__()
        assert nFrames % 2 == 1 and nFrames >= 1
        self.root = root_dir
        self.ids = list(split_ids)
        self.tt = nFrames // 2
        self.augment = augment
        self.patch = patch_size
        self.cbct_window = cbct_window # <--- Modification
        self.ct_window = ct_window   # <--- Modification
        self.min_hw = int(min_hw)
        self.verbose = verbose

        self.cases = []
        self.case_stats = []
        self.dropped_cases = []

        for pid in self.ids:
            case_dir = join(self.root, pid)
            cbct_path = join(case_dir, "cbct.nii.gz")
            ct_path = join(case_dir, "ct.nii.gz")
            mask_path = join(case_dir, "mask.nii.gz")

            if not os.path.exists(cbct_path) or not os.path.exists(ct_path):
                raise FileNotFoundError(f"{pid} is missing cbct.nii.gz or ct.nii.gz")

            cbct_raw = load_nii(cbct_path)
            ct_raw = load_nii(ct_path)

            # --- Core Modification: Call single window width normalization function ---
            cbct = normalize_hu_single_window(cbct_raw, self.cbct_window)  # Shape: [1, Z, H, W]
            ct = normalize_hu_single_window(ct_raw, self.ct_window)      # Shape: [1, Z, H, W]

            # Apply mask to CT
            if os.path.exists(mask_path):
                mask = load_nii(mask_path).astype(np.float32)
                mask = (mask > 0).astype(np.float32)
            else:
                if self.verbose: print(f"[Warning] {pid} mask.nii.gz not found, using CBCT non-zero region instead")
                mask = (cbct_raw > 0).astype(np.float32)

            ct = ct * mask[None, ...]  # Expand mask dimension for broadcasting

            if cbct.shape[1:] != ct.shape[1:]:
                raise ValueError(f"{pid}: CBCT and CT voxel dimensions do not match")

            _, Z, H, W = cbct.shape

            if self.min_hw > 0 and (H < self.min_hw or W < self.min_hw):
                self.dropped_cases.append((pid, Z, H, W))
                continue

            self.cases.append((pid, cbct, ct, Z))
            full_neighbors = max(0, Z - 2 * self.tt)
            self.case_stats.append({"pid": pid, "Z": Z, "H": H, "W": W, "full_neighbors": full_neighbors})

        self.sample_index = []
        for ci, (_, _, _, Z) in enumerate(self.cases):
            for z in range(Z): self.sample_index.append((ci, z))

        if self.verbose: self._print_stats()

    def _print_stats(self):
        print("==== CBCT2SCTNiftiDataset Statistics ====")
        if self.dropped_cases:
            print(f"Cases dropped by min_hw={self.min_hw}: {len(self.dropped_cases)}")
            for pid, Z, H, W in self.dropped_cases:
                print(f"  - {pid}: Z={Z}, H={H}, W={W}  -> Dropped")

        print(f"Valid cases count: {len(self.cases)}")
        print("Total slices (including boundary centers):", len(self.sample_index))
        print("===========================================")

    def __len__(self):
        return len(self.sample_index)

    @staticmethod
    def _rand_flip_rot(a: np.ndarray) -> np.ndarray:
        # Augmentation applicable to single/multi-channel [C,H,W] arrays
        if np.random.rand() < 0.5: a = np.flip(a, axis=1)  # Vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2)  # Horizontal flip
        if np.random.rand() < 0.5: a = np.rot90(a, 2, axes=(1, 2))  # 180° rotation
        return a

    @staticmethod
    def _rand_patch(a: np.ndarray, ps: int):
        C, H, W = a.shape
        if ps <= 0 or ps > min(H, W): return a, 0, 0
        y = np.random.randint(0, H - ps + 1)
        x = np.random.randint(0, W - ps + 1)
        return a[:, y:y + ps, x:x + ps], y, x

    def _neighbors(self, vol: np.ndarray, zc: int) -> List[np.ndarray]:
        C, Z, H, W = vol.shape
        offsets = [o for o in range(-self.tt, self.tt + 1) if o != 0]
        nbs = [vol[:, clamp_index(zc + o, 0, Z - 1)] for o in offsets]
        return nbs

    def __getitem__(self, idx):
        ci, zc = self.sample_index[idx]
        pid, cbct, ct, Z = self.cases[ci]

        # Slice is already single-channel [1, H, W]
        in_c = cbct[:, zc]
        gt_c = ct[:, zc]
        nbs = self._neighbors(cbct, zc)

        if self.augment:
            in_c = self._rand_flip_rot(in_c)
            gt_c = self._rand_flip_rot(gt_c)
            nbs = [self._rand_flip_rot(n) for n in nbs]

        if self.patch > 0:
            in_c, y, x = self._rand_patch(in_c, self.patch)
            gt_c = gt_c[:, y:y + self.patch, x:x + self.patch]
            nbs = [n[:, y:y + self.patch, x:x + self.patch] for n in nbs]

        # Convert to Tensor
        in_t = torch.from_numpy(in_c.copy())
        gt_t = torch.from_numpy(gt_c.copy())
        nn_t = [torch.from_numpy(n.copy()) for n in nbs]
        bic_t = in_t  # Placeholder

        return gt_t.float(), in_t.float(), nn_t, bic_t.float()


# =============================================================================
# == Test Entry (This file can be run directly for testing)
# =============================================================================
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    # --- Core Modification: Define single window width ---
    DEFAULT_CBCT_WINDOW = (-1500, 2500)
    DEFAULT_CT_WINDOW = (-1000, 1500)

    parser = argparse.ArgumentParser("dataset_cbct2sct_nifti (Single Window Width Version) Test")
    parser.add_argument("--root_dir", type=str, default="./dataset/Task2/brain")  # Please modify to your data path
    parser.add_argument("--ids", type=str, nargs="*", default=[], help="Specify test case IDs")
    parser.add_argument("--nframes", type=int, default=5)
    parser.add_argument("--patch", type=int, default=192)
    parser.add_argument("--save_dir", type=str, default="vis_single_window")
    args = parser.parse_args()

    if not args.ids:
        all_ids = [d for d in os.listdir(args.root_dir) if os.path.isdir(join(args.root_dir, d))]
        args.ids = sorted(all_ids)[:1]  # Test only the first case by default
    print("Test cases:", args.ids)

    ds = CBCT2SCTNiftiDataset(
        root_dir=args.root_dir,
        split_ids=args.ids,
        cbct_window=DEFAULT_CBCT_WINDOW,  # <--- Modification
        ct_window=DEFAULT_CT_WINDOW,    # <--- Modification
        nFrames=args.nframes,
        augment=False,
        patch_size=args.patch,
        verbose=True,
    )
    print("Total samples:", len(ds))

    gt_t, inp_t, nbs_t, _ = ds[len(ds) // 2]  # Take middle sample
    print(f"Single sample shapes: GT {gt_t.shape}, Input {inp_t.shape}, Neighbors {len(nbs_t)}x{nbs_t[0].shape}")

    # --- Visualization ---
    os.makedirs(args.save_dir, exist_ok=True)
    # --- Core Modification: Visualize one CBCT and one CT image each ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(inp_t.squeeze(), cmap='gray') # squeeze() removes dimension with size 1
    ax1.set_title(f'Input (CBCT)\nWindow: {DEFAULT_CBCT_WINDOW}')
    ax1.axis('off')

    ax2.imshow(gt_t.squeeze(), cmap='gray')
    ax2.set_title(f'GT (CT)\nWindow: {DEFAULT_CT_WINDOW}')
    ax2.axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, f"single_window_sample_{args.ids[0]}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Visualized sample saved to: {save_path}")
    plt.show()