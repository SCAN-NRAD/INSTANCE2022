import argparse
import pickle
import sys
from functools import partial
from typing import List

import haiku as hk
import nibabel as nib
import numpy as np

import jax


def ite(total: int, size: int, pad: int) -> List[int]:
    r"""
    Generate a list of patch indices such that the center of the patches (unpaded patches) cover the full image.

    Args:
        total: The total size of the image.
        size: The size of the patch.
        pad: The padding of the patch.
    """
    naive = list(range(0, total - size, size - 2 * pad)) + [total - size]
    return np.round(np.linspace(0, total - size, len(naive))).astype(int)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--path", type=str, default=".", help="Path to wandb run")
    parser.add_argument("--data", type=str, default=".", help="Path to data")
    parser.add_argument("--indices", nargs="+", type=int, required=True, help="Indices of the runs to evaluate")
    args = parser.parse_args()

    print(args.path)
    sys.path.insert(0, args.path)
    import model  # noqa: F401
    from functions import load_miccai22, unpad  # noqa: F401

    @partial(jax.jit, static_argnums=(2,))
    def f(w, x, zooms):
        return hk.without_apply_rng(hk.transform(model.unet_with_groups)).apply(w, x, zooms)

    w = pickle.load(open(f"{args.path}/w.pkl", "rb"))

    for idx in args.indices:
        print(f"Evaluating run {idx}", end=" ", flush=True)
        img, lab, zooms = load_miccai22(args.data, idx)

        size = (100, 100, 25)
        pads = (16, 16, 1)

        sum = np.zeros_like(img)
        num = np.zeros_like(img)

        for i in ite(img.shape[0], size[0], pads[0]):
            for j in ite(img.shape[1], size[1], pads[1]):
                for k in ite(img.shape[2], size[2], pads[2]):
                    x = img[i : i + size[0], j : j + size[1], k : k + size[2]]
                    p = f(w, x, zooms)
                    p = unpad(p, pads)

                    sum[
                        i + pads[0] : i + size[0] - pads[0],
                        j + pads[1] : j + size[1] - pads[1],
                        k + pads[2] : k + size[2] - pads[2],
                    ] += p
                    num[
                        i + pads[0] : i + size[0] - pads[0],
                        j + pads[1] : j + size[1] - pads[1],
                        k + pads[2] : k + size[2] - pads[2],
                    ] += 1.0

                    print(".", end="", flush=True)
        print(flush=True)

        negative_value = -10.0
        sum[num == 0] = negative_value
        num[num == 0] = 1.0

        pred = sum / num

        original = nib.load(f"{args.data}/label/{idx:03d}.nii.gz")

        img = nib.Nifti1Image(pred, original.affine, original.header)
        nib.save(img, f"{args.path}/eval{idx:03d}.nii.gz")

        lab = (np.sign(pred) + 1) / 2
        lab = 2 * original.get_fdata() + lab
        img = nib.Nifti1Image(lab, original.affine, original.header)
        nib.save(img, f"{args.path}/confusion{idx:03d}.nii.gz")


if __name__ == "__main__":
    main()
