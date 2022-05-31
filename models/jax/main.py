import argparse
import shutil
import time
from textwrap import dedent

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax.config import config
from sklearn.metrics import confusion_matrix
from functools import partial

from functions import format_time, load_miccai22, random_sample, unpad, cross_entropy
from model import unet_with_groups

config.update("jax_debug_nans", True)
config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def main():
    print("main", flush=True)
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data", type=str, default=".", help="Path to data")
    parser.add_argument("--logdir", type=str, default=".", help="Path to log directory")
    parser.add_argument("--seed_init", type=int, default=1, help="Random seed")
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to npy file")
    args = parser.parse_args()

    wandb.init(project="miccai22", name=args.name, dir=args.logdir)
    shutil.copy(__file__, f"{wandb.run.dir}/main.py")
    shutil.copy("./model.py", f"{wandb.run.dir}/model.py")
    shutil.copy("./functions.py", f"{wandb.run.dir}/functions.py")
    shutil.copy("./evaluate.py", f"{wandb.run.dir}/evaluate.py")

    # Load data
    print("Loading data...", flush=True)
    img, lab, zooms = load_miccai22(args.data, 1)

    # Create model
    model = hk.without_apply_rng(hk.transform(unet_with_groups))

    if args.pretrained is not None:
        print("Loading pretrained parameters...", flush=True)
        w = np.load(args.pretrained)
    else:
        print("Initializing model...", flush=True)
        t = time.perf_counter()
        w = model.init(jax.random.PRNGKey(args.seed_init), img[:16, :16, :16], zooms)
        print(f"Initialized model in {format_time(time.perf_counter() - t)}", flush=True)

    opt_state = optax.adam(args.lr).init(w)

    def un(img):
        return unpad(img, (16, 16, 1))

    @partial(jax.jit, static_argnums=(4,))
    def update(w, opt_state, x, y, zooms, lr):
        r"""Update the model parameters.

        Args:
            w: Model parameters.
            opt_state: Optimizer state.
            x: Input data ``(x, y, z)``.
            y: Ground truth data ``(x, y, z)`` of the form (-1.0, 1.0).
            zooms: The zoom factors ``(x, y, z)``.
            lr: Learning rate.

        Returns:
            (w, opt_state, loss, pred):
                w: Updated model parameters.
                opt_state: Updated optimizer state.
                loss: Loss value.
                pred: Predicted data ``(x, y, z)``.
        """
        assert x.ndim == 3
        assert y.ndim == 3

        def h(w, x, y):
            p = model.apply(w, x, zooms)
            return jnp.mean(cross_entropy(un(p), un(y))), p

        grad_fn = jax.value_and_grad(h, has_aux=True)
        (loss, pred), grads = grad_fn(w, x, y)

        updates, opt_state = optax.adam(lr).update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        return w, opt_state, loss, pred

    time0 = time.perf_counter()
    confusion_matrices = []

    for i in range(144_001):
        ok = True
        if ok:
            ldict = {
                "i": i,
                "w": w,
                "opt_state": opt_state,
                "time0": time0,
                "args": args,
                "update": update,
                "confusion_matrices": confusion_matrices,
                "un": un,
            }
            for trial in range(10):
                try:
                    code = dedent(open(f"{wandb.run.dir}/main.py", "r").read().split("# Loop")[2])
                    exec(code, globals(), ldict)
                    ok = True
                    break
                except Exception as e:
                    ok = False
                    print(e)
                    print(f"Trial {trial} failed, retrying again in 1min...", flush=True)
                    time.sleep(60)
            if not ok:
                raise Exception("Failed to run loop in 10 trials, aborting")
            w = ldict["w"]
            confusion_matrices = ldict["confusion_matrices"]
        else:
            # Loop
            if i == 20:
                jax.profiler.start_trace(wandb.run.dir)

            img, lab, zooms = load_miccai22(args.data, 1 + i % 90)

            # regroup zooms and sizes by rounding and taking subsets of the volume
            zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
            sample_size = (100, 100, 25)  # physical size ~= 50mm x 50mm x 125mm
            if np.random.rand() < 0.5:
                while True:
                    x, y = random_sample(img, lab, sample_size)
                    if np.any(un(y) == 1):
                        img, lab = x, y
                        break
            else:
                img, lab = random_sample(img, lab, sample_size)

            t0 = time.perf_counter()

            w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, args.lr)
            train_loss.block_until_ready()

            confusion_matrices.append(confusion_matrix(lab.flatten() > 0.0, train_pred.flatten() > 0.0))
            epoch_avg_confusion = np.mean(confusion_matrices[-90:], axis=0)
            epoch_avg_confusion = epoch_avg_confusion / np.sum(epoch_avg_confusion)

            print(
                (
                    f"{wandb.run.dir.split('/')[-2]} "
                    f"[{i + 1:04d}:{format_time(time.perf_counter() - time0)}] "
                    f"train[ loss={train_loss:.2f} "
                    f"1={np.sum(lab == 1)} "
                    f"tn={epoch_avg_confusion[0, 0]:.2f} tp={epoch_avg_confusion[1, 1]:.2f} "
                    f"fn={epoch_avg_confusion[1, 0]:.2f} fp={epoch_avg_confusion[0, 1]:.2f} "
                    f"min-median-max={np.min(train_pred):.2f}/{np.median(train_pred):.2f}/{np.max(train_pred):.2f} "
                    f"update_time={format_time(time.perf_counter() - t0)} "
                ),
                flush=True,
            )

            state = {
                "iteration": i,
                "_runtime": time.perf_counter() - time0,
                "train_loss": train_loss,
                "true_negatives": epoch_avg_confusion[0, 0],
                "true_positives": epoch_avg_confusion[1, 1],
                "false_negatives": epoch_avg_confusion[1, 0],
                "false_positives": epoch_avg_confusion[0, 1],
            }

            if i % 500 == 0:
                # save model using wandb and numpy
                np.save(f"{wandb.run.dir}/w.npy", w)

            if i == 20:
                jax.profiler.stop_trace()

            wandb.log(state)
            # Loop


if __name__ == "__main__":
    main()
