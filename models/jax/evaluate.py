import argparse
import pickle
import sys
from functools import partial

import haiku as hk
import nibabel as nib
import numpy as np
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

import jax


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--path", type=str, default=".", help="Path to wandb run")
    parser.add_argument("--weights", type=str, default="w.pkl", help="Relative path to weights")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for segmentation")
    parser.add_argument("--data", type=str, default=".", help="Path to data")
    parser.add_argument("--indices", nargs="+", type=int, required=True, help="Indices of the runs to evaluate")
    args = parser.parse_args()

    print(args.path, flush=True)
    sys.path.insert(0, args.path)
    import model  # noqa: F401
    from functions import load_miccai22, round_zooms, eval_model  # noqa: F401

    # Load model args
    with open(f"{args.path}/args.pkl", "rb") as f:
        train_args = pickle.load(f)
    print(train_args, flush=True)

    @partial(jax.jit, static_argnums=(2,))
    def apply(w, x, zooms):
        return hk.without_apply_rng(hk.transform(model.unet_with_groups(train_args))).apply(w, x, zooms)

    with open(f"{args.path}/{args.weights}", "rb") as f:
        w = pickle.load(f)

    collect_metrics = []

    for idx in args.indices:
        print(f"Evaluating run {idx}", flush=True)
        img, lab, zooms = load_miccai22(args.data, idx)
        zooms = round_zooms(zooms)
        pred = eval_model(img, lambda x: apply(w, x, zooms), overlap=2.0, verbose=True)
        print(flush=True)

        original = nib.load(f"{args.data}/label/{idx:03d}.nii.gz")
        y_gt = original.get_fdata()

        img = nib.Nifti1Image(pred, original.affine, original.header)
        nib.save(img, f"{args.path}/eval{idx:03d}.nii.gz")

        y_pred = (np.sign(pred - args.threshold) + 1) / 2
        four_classes = 2 * y_gt + y_pred
        img = nib.Nifti1Image(four_classes, original.affine, original.header)
        nib.save(img, f"{args.path}/confusion{idx:03d}.nii.gz")

        tp = np.sum(y_pred * y_gt)
        # tn = np.sum((1 - y_pred) * (1 - y_gt))
        fp = np.sum(y_pred * (1 - y_gt))
        fn = np.sum((1 - y_pred) * y_gt)

        DSC = 2 * tp / (2 * tp + fp + fn)
        print(f"DSC (dice score): {DSC}", flush=True)

        HD = compute_hausdorff_distance(y_pred[None, None], y_gt[None, None]).item()
        print(f"HD (hausdorff distance): {HD}", flush=True)

        RVD = (tp + fp) / (tp + fn) - 1
        print(f"RVD (relative volume difference): {RVD}", flush=True)

        collect_metrics.append([DSC, HD, RVD])

    collect_metrics = np.array(collect_metrics)

    print(f"DSC (dice score):                 {', '.join(map(repr, collect_metrics[:,0]))}")
    print(f"HD (hausdorff distance):          {', '.join(map(repr, collect_metrics[:,1]))}")
    print(f"RVD (relative volume difference): {', '.join(map(repr, collect_metrics[:,2]))}")

    avg_metrics = np.mean(collect_metrics, axis=0)
    print(f"Average DSC: {avg_metrics[0]}", flush=True)
    print(f"Average HD: {avg_metrics[1]}", flush=True)
    print(f"Average RVD: {avg_metrics[2]}", flush=True)


if __name__ == "__main__":
    main()
