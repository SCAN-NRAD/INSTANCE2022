import argparse
import importlib
import pickle
import shutil
import sys
import time
from functools import partial

import haiku as hk
import numpy as np
import optax
import hashlib

import jax
import jax.numpy as jnp
import wandb
from jax.config import config
from model import unet_with_groups

config.update("jax_debug_nans", True)
config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    print("main", flush=True)
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data", type=str, default="../../train_2", help="Path to data")
    parser.add_argument("--logdir", type=str, default=".", help="Path to log directory")
    parser.add_argument("--seed_init", type=int, default=1, help="Random seed")
    parser.add_argument("--name", type=str, required=True, help="Name of the run")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to npy file")
    parser.add_argument("--equivariance", type=str, default="E3", help="Equivariance group")
    parser.add_argument("--width", type=int, default=5, help="Width of the network")
    parser.add_argument(
        "--num_radial_basis_sh0", type=int, default=2, help="Number of radial basis functions for spherical harmonics 0"
    )
    parser.add_argument(
        "--num_radial_basis_sh1", type=int, default=2, help="Number of radial basis functions for spherical harmonics 1"
    )
    parser.add_argument(
        "--num_radial_basis_sh2", type=int, default=2, help="Number of radial basis functions for spherical harmonics 2"
    )
    parser.add_argument(
        "--num_radial_basis_sh3", type=int, default=0, help="Number of radial basis functions for spherical harmonics 3"
    )
    parser.add_argument(
        "--relative_start_sh2", type=float, default=0.0, help="Relative start of radial basis for spherical harmonics 2"
    )
    parser.add_argument(
        "--relative_start_sh3", type=float, default=0.0, help="Relative start of radial basis for spherical harmonics 3"
    )
    parser.add_argument("--min_zoom", type=float, default=0.36, help="Minimum zoom")
    parser.add_argument("--downsampling", type=float, default=2.0, help="Downsampling factor")
    parser.add_argument("--conv_diameter", type=float, default=5.0, help="Diameter of the convolution kernel")
    parser.add_argument("--dummy", type=int, default=0, help="Dummy model to test code")
    args = parser.parse_args()

    wandb.init(project="miccai22", name=args.name, dir=args.logdir, config=args)
    shutil.copy(__file__, f"{wandb.run.dir}/main.py")
    shutil.copy("./model.py", f"{wandb.run.dir}/model.py")
    shutil.copy("./functions.py", f"{wandb.run.dir}/functions.py")
    shutil.copy("./evaluate.py", f"{wandb.run.dir}/evaluate.py")
    sys.path.insert(0, wandb.run.dir)
    import functions

    with open(f"{wandb.run.dir}/args.pkl", "wb") as f:
        pickle.dump(args, f)

    # Load data
    print("Loading data...", flush=True)
    img, lab, zooms = functions.load_miccai22(args.data, 1)

    # Create model
    model = hk.without_apply_rng(hk.transform(unet_with_groups(args)))

    if args.pretrained is not None:
        print("Loading pretrained parameters...", flush=True)
        w = pickle.load(open(args.pretrained, "rb"))
    else:
        print("Initializing model...", flush=True)
        t = time.perf_counter()
        w = model.init(jax.random.PRNGKey(args.seed_init), img[:100, :100, :25], zooms)
        print(f"Initialized model in {functions.format_time(time.perf_counter() - t)}", flush=True)

    def un(img):
        return functions.unpad(img, (16, 16, 1))

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
            return jnp.mean(functions.cross_entropy(un(p), un(y))), p

        grad_fn = jax.value_and_grad(h, has_aux=True)
        (loss, pred), grads = grad_fn(w, x, y)

        updates, opt_state = optax.adam(lr).update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        return w, opt_state, loss, pred

    opt_state = optax.adam(args.lr).init(w)

    hash = hash_file(f"{wandb.run.dir}/functions.py")
    state = functions.init_train_loop(args, w, opt_state)

    for i in range(99_999_999):

        # Reload the loop function if the code has changed
        new_hash = hash_file(f"{wandb.run.dir}/functions.py")
        if new_hash != hash:
            hash = new_hash
            importlib.reload(functions)
            state = functions.init_train_loop(args, w, opt_state)

        state, w, opt_state = functions.train_loop(args, state, i, w, opt_state, un, update, apply_model)


if __name__ == "__main__":
    main()
