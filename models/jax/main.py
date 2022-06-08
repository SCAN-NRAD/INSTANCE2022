import argparse
import pickle
import shutil
import time
from functools import partial
from textwrap import dedent

import haiku as hk
import numpy as np
import optax
from sklearn.metrics import confusion_matrix

import jax
import jax.numpy as jnp
import wandb
from functions import cross_entropy, format_time, load_miccai22, random_sample, unpad
from jax.config import config
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
    parser.add_argument("--data", type=str, default="../../train_2", help="Path to data")
    parser.add_argument("--logdir", type=str, default=".", help="Path to log directory")
    parser.add_argument("--seed_init", type=int, default=1, help="Random seed")
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to npy file")
    parser.add_argument("--equivariance", type=str, default="E3", help="Equivariance group")
    parser.add_argument("--width", type=int, default=5, help="Width of the network")
    parser.add_argument("--num_radial_basis", type=int, default=2, help="Number of radial basis functions")
    parser.add_argument("--min_zoom", type=float, default=0.36, help="Minimum zoom")
    parser.add_argument("--downsampling", type=float, default=2.0, help="Downsampling factor")
    parser.add_argument("--dummy", type=int, default=0, help="Dummy model to test code")
    args = parser.parse_args()

    wandb.init(project="miccai22", name=args.name, dir=args.logdir, config=args)
    shutil.copy(__file__, f"{wandb.run.dir}/main.py")
    shutil.copy("./model.py", f"{wandb.run.dir}/model.py")
    shutil.copy("./functions.py", f"{wandb.run.dir}/functions.py")
    shutil.copy("./evaluate.py", f"{wandb.run.dir}/evaluate.py")

    with open(f"{wandb.run.dir}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    # Load data
    print("Loading data...", flush=True)
    img, lab, zooms = load_miccai22(args.data, 1)

    # Create model
    if args.dummy:
        model = hk.without_apply_rng(
            hk.transform(lambda x, zooms: jax.vmap(jax.vmap(jax.vmap(hk.Linear(1))))(x[..., None])[..., 0])
        )
    else:
        model = hk.without_apply_rng(hk.transform(unet_with_groups(args)))

    if args.pretrained is not None:
        print("Loading pretrained parameters...", flush=True)
        w = pickle.load(open(args.pretrained, "rb"))
    else:
        print("Initializing model...", flush=True)
        t = time.perf_counter()
        w = model.init(jax.random.PRNGKey(args.seed_init), img[:16, :16, :16], zooms)
        print(f"Initialized model in {format_time(time.perf_counter() - t)}", flush=True)

    opt_state = optax.adam(args.lr).init(w)

    def un(img):
        return unpad(img, (16, 16, 1))

    @partial(jax.jit, static_argnums=(2,))
    def apply_model(w, x, zooms):
        return model.apply(w, x, zooms)

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

    last_code = None
    var = locals()

    # Init
    print("Init statistics...", flush=True)
    time0 = time.perf_counter()
    confusion_matrices = []
    sample_size = (100, 100, 25)  # physical size ~= 50mm x 50mm x 125mm

    test_set = []
    for i in range(91, 100 + 1):
        img, lab, zooms = load_miccai22(args.data, i)  # test data
        zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
        center_of_mass = np.stack(np.nonzero(lab == 1.0), axis=-1).mean(0).astype(np.int)
        start = np.maximum(center_of_mass - np.array(sample_size) // 2, 0)
        end = np.minimum(start + np.array(sample_size), np.array(img.shape))
        start = end - np.array(sample_size)
        img = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        lab = lab[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        test_set.append((img, lab, zooms))

    t4 = time.perf_counter()
    # Init

    for i in range(144_001):
        var["i"] = i

        ok = True
        if ok:
            for trial in range(10):
                try:
                    with open(f"{wandb.run.dir}/main.py", "r") as f:
                        code = dedent(f.read().split("# Loop")[3])
                    if code != last_code:
                        with open(f"{wandb.run.dir}/main.py", "r") as f:
                            code = dedent(f.read().split("# Init")[1])
                        exec(code, globals(), var)
                        with open(f"{wandb.run.dir}/main.py", "r") as f:
                            code = dedent(f.read().split("# Loop")[3])
                        last_code = code
                    exec(code, globals(), var)
                    ok = True
                    break
                except Exception as e:
                    ok = False
                    print(f"Trial {trial} failed", flush=True)
                    print(e)
                    print("\nRetrying again in 1min...", flush=True)
                    time.sleep(60)
            if not ok:
                raise Exception("Failed to run loop in 10 trials, aborting")
        else:
            # Loop
            t0 = time.perf_counter()
            t_extra = t0 - t4

            if i == 120:
                jax.profiler.start_trace(wandb.run.dir)

            img, lab, zooms = load_miccai22(args.data, 1 + i % 90)
            t01 = time.perf_counter()

            # regroup zooms and sizes by rounding and taking subsets of the volume
            zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
            if np.random.rand() < 0.5:
                while True:
                    x, y = random_sample(img, lab, sample_size)
                    if np.any(un(y) == 1):
                        img, lab = x, y
                        break
            else:
                img, lab = random_sample(img, lab, sample_size)

            t1 = time.perf_counter()

            w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, args.lr)
            train_loss.block_until_ready()

            t2 = time.perf_counter()
            img, lab, zooms = test_set[i % 10]
            test_pred = apply_model(w, img, zooms)
            t3 = time.perf_counter()
            confusion_matrices.append(confusion_matrix(un(lab).flatten() > 0.0, un(test_pred).flatten() > 0.0))
            epoch_avg_confusion = np.mean(confusion_matrices[-10:], axis=0)
            epoch_avg_confusion = epoch_avg_confusion / np.sum(epoch_avg_confusion)

            min_median_max = np.min(train_pred), np.median(train_pred), np.max(train_pred)
            t4 = time.perf_counter()

            print(
                (
                    f"{wandb.run.dir.split('/')[-2]} "
                    f"[{i + 1:04d}:{format_time(time.perf_counter() - time0)}] "
                    f"train[ loss={train_loss:.3f} "
                    f"tn={epoch_avg_confusion[0, 0]:.2f} tp={epoch_avg_confusion[1, 1]:.2f} "
                    f"fn={epoch_avg_confusion[1, 0]:.2f} fp={epoch_avg_confusion[0, 1]:.2f} "
                    f"min-median-max={min_median_max[0]:.2f} {min_median_max[1]:.2f} {min_median_max[2]:.2f} ] "
                    f"time="
                    f"L{format_time(t01 - t0)}+S{format_time(t1 - t01)}"
                    f"U{format_time(t2 - t1)}+"
                    f"E{format_time(t3 - t2)}+C{format_time(t4 - t3)}+"
                    f"EX{format_time(t_extra)} "
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
                "min_pred": min_median_max[0],
                "median_pred": min_median_max[1],
                "max_pred": min_median_max[2],
            }

            if i % 500 == 0:
                with open(f"{wandb.run.dir}/w.pkl", "wb") as f:
                    pickle.dump(w, f)

            if i == 120:
                jax.profiler.stop_trace()

            wandb.log(state)
            # Loop


if __name__ == "__main__":
    main()
