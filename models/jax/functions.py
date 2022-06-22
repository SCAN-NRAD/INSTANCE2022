import math
import pickle
import random
import time
from collections import namedtuple
from typing import Callable, List, Tuple

import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp
import wandb
from diffeomorphism import deform, scalar_field
from jax.scipy.special import logsumexp


@jax.jit
def noise_mri(rng, img):
    assert img.shape[0] == img.shape[1]
    noise = jax.vmap(lambda r: scalar_field(img.shape[0], 128, r), 0, 2)(jax.random.split(rng, img.shape[2] * img.shape[3]))
    return img + 1e-2 * jnp.reshape(noise, img.shape)


@jax.jit
def deform_mri(rng, img, temperature):
    r"""Apply a diffeomorphism to an MRI image.
    Apply the same deformation to all slices of the image.

    Args:
        rng: A random number generator.
        img: The MRI image ``[x, y, ...]``

    Returns:
        The deformed image.
    """
    assert img.shape[0] == img.shape[1]
    f = lambda i: deform(i, temperature, 5, rng, "nearest")
    for _ in range(img.ndim - 2):
        f = jax.vmap(f, -1, -1)
    return f(img)


def load_miccai22(path: str, i: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """i goes from 1 to 100"""
    image = nib.load(f"{path}/data/{i:03d}.nii.gz")
    label = nib.load(f"{path}/label/{i:03d}.nii.gz")
    assert np.allclose(image.affine, label.affine)

    zooms = image.header.get_zooms()
    image = image.get_fdata()
    label = label.get_fdata()

    bone = 0.005 * np.clip(image, a_min=0.0, a_max=1000.0).astype(np.float32)
    range1 = 0.03 * np.clip(image, a_min=0.0, a_max=80.0).astype(np.float32)
    range2 = 0.013 * np.clip(image, a_min=-50.0, a_max=220.0).astype(np.float32)
    image = np.stack([bone, range1, range2], axis=-1)

    label = 2.0 * label - 1.0
    label = label.astype(np.float32)
    return image, label, zooms


def random_slice(size: int, target_size: int) -> slice:
    if size <= target_size:
        return slice(None)
    start = random.randint(0, size - target_size)
    return slice(start, start + target_size)


def random_sample(x: np.ndarray, y: np.ndarray, target_sizes: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    slices = jax.tree_map(random_slice, x.shape[:3], target_sizes)
    sx = x[slices[0], slices[1], slices[2]]
    sy = y[slices[0], slices[1], slices[2]]
    return sx, sy


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{1000 * seconds:03.0f}ms"
    if seconds < 60:
        return f"{seconds:05.2f}s"
    minutes = math.floor(seconds / 60)
    seconds = seconds - 60 * minutes
    if minutes < 60:
        return f"{minutes:02.0f}min{seconds:02.0f}s"
    hours = math.floor(minutes / 60)
    minutes = minutes - 60 * hours
    return f"{hours:.0f}h{minutes:02.0f}min"


def unpad(x, pads):
    return x[pads[0] : -pads[0], pads[1] : -pads[1], pads[2] : -pads[2]]


def cross_entropy(p: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r"""Safe cross entropy loss function.

    Args:
        p: predictions (logits)
        y: labels (-1 or 1)

    Returns:
        loss: cross entropy loss ``log(1 + exp(-p y))``
    """
    assert p.shape == y.shape
    zero = jnp.zeros_like(p)
    return logsumexp(jnp.stack([zero, -p * y]), axis=0)


@jax.jit
def confusion_matrix(y: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    r"""Confusion matrix.

    Args:
        y: labels (-1 or 1)
        p: predictions (logits)

    Returns:
        confusion matrix
        [tn, fp]
        [fn, tp]
    """
    assert y.shape == p.shape
    y = y > 0.0
    p = p > 0.0
    tp = jnp.sum(y * p)
    tn = jnp.sum((1 - y) * (1 - p))
    fp = jnp.sum((1 - y) * p)
    fn = jnp.sum(y * (1 - p))
    return jnp.array([[tn, fp], [fn, tp]])


TrainState = namedtuple(
    "TrainState",
    [
        "time0",
        "train_set",
        "test_set",
        "t4",
        "losses",
        "best_sorted_dices",
    ],
)


def init_train_loop(args, old_state, step, w, opt_state) -> TrainState:
    print("Prepare for the training loop...", flush=True)

    if args.dummy:
        train_idx = [1, 2]
        test_idx = [91, 92]
    else:
        train_idx = list(range(args.trainset_start, args.trainset_stop + 1))
        test_idx = [i for i in list(range(1, 100 + 1)) if i not in train_idx]

    train_set = [load_miccai22(args.data, i) for i in train_idx]

    test_set = []
    test_sample_size = np.array([200, 200, 25])
    for i in test_idx:
        img, lab, zooms = load_miccai22(args.data, i)  # test data
        zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
        center_of_mass = np.stack(np.nonzero(lab == 1.0), axis=-1).mean(0).astype(np.int)
        start = np.maximum(center_of_mass - test_sample_size // 2, 0)
        end = np.minimum(start + test_sample_size, np.array(img.shape[:3]))
        start = end - test_sample_size
        img = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        lab = lab[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        test_set.append((img, lab, zooms))

    return TrainState(
        time0=getattr(old_state, "time0", time.perf_counter()),
        train_set=train_set,
        test_set=test_set,
        t4=time.perf_counter(),
        losses=getattr(old_state, "losses", np.ones((len(train_set),))),
        best_sorted_dices=getattr(old_state, "best_sorted_dices", np.zeros((len(test_set),))),
    )


sample_size = (144, 144, 13)  # physical size ~= 65mm ^ 3
sample_padding = (22, 22, 2)  # 10mm of border removed


def train_loop(args, state: TrainState, step, w, opt_state, update, apply_model) -> TrainState:
    t0 = time.perf_counter()
    t_extra = t0 - state.t4

    if step == 120:
        jax.profiler.start_trace(wandb.run.dir)

    img, lab, zooms = state.train_set[step % len(state.train_set)]

    # data augmentation
    rng = jax.random.split(jax.random.PRNGKey(step), 4)

    if jax.random.uniform(rng[0]) < args.augmentation_noise:
        img = noise_mri(rng[1], img)

    if jax.random.uniform(rng[2]) < args.augmentation_deformation:
        img = deform_mri(rng[3], img, args.deformation_temperature)
        lab = deform_mri(rng[3], lab[..., np.newaxis], args.deformation_temperature)[..., 0]
        lab = jnp.round(lab)

    # regroup zooms and sizes by rounding and taking subsets of the volume
    zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
    if np.random.rand() < 0.5:
        # avoid patch without label
        while True:
            x, y = random_sample(img, lab, sample_size)
            if np.any(unpad(y, sample_padding) == 1):
                img, lab = x, y
                break
    else:
        # avoid patch full of air
        while True:
            x, y = random_sample(img, lab, sample_size)
            if np.any(x > 0.0):
                img, lab = x, y
                break
    del x, y

    t1 = time.perf_counter()

    lr = args.lr * max(0.1 ** math.floor(step / args.lr_div_step), 0.1)
    w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, lr, sample_padding)
    train_loss.block_until_ready()

    t2 = time.perf_counter()
    c = np.array(confusion_matrix(unpad(lab, sample_padding), unpad(train_pred, sample_padding)))
    with np.errstate(invalid="ignore"):
        train_dice = 2 * c[1, 1] / (2 * c[1, 1] + c[1, 0] + c[0, 1])

    min_median_max = np.min(train_pred), np.median(train_pred), np.max(train_pred)

    state.losses[step % len(state.train_set)] = train_loss

    time_str = time.strftime("%H:%M", time.localtime())
    print(
        (
            f"{wandb.run.dir.split('/')[-2]} "
            f"[{time_str}] "
            f"[{step:04d}:{format_time(time.perf_counter() - state.time0)}] "
            f"train[ loss={np.mean(state.losses):.4f} "
            f"LR={lr:.1e} "
            f"dice={100 * train_dice:02.0f} ] "
            f"time[ "
            f"S{format_time(t1 - t0)}+"
            f"U{format_time(t2 - t1)}+"
            f"EX{format_time(t_extra)} ]"
        ),
        flush=True,
    )

    if step % 500 == 0:
        with open(f"{wandb.run.dir}/w.pkl", "wb") as f:
            pickle.dump(w, f)

    if step == 120:
        jax.profiler.stop_trace()

    if step % 100 == 0:
        c = np.zeros((len(state.test_set), 2, 2))
        for j, (img, lab, zooms) in enumerate(state.test_set):
            test_pred = eval_model(
                img,
                lambda x: apply_model(w, x, zooms),
                overlap=1.0,
            )
            c[j] = np.array(confusion_matrix(lab, test_pred))

        with np.errstate(invalid="ignore"):
            dice = 2 * c[:, 1, 1] / (2 * c[:, 1, 1] + c[:, 1, 0] + c[:, 0, 1])

        wandb.log({f"dice_{91 + i}": d for i, d in enumerate(dice)}, commit=False, step=step)

        sorted_dices = np.sort(dice)
        for i, (old_dice, new_dice) in enumerate(zip(state.best_sorted_dices, sorted_dices)):
            if new_dice > old_dice:
                state.best_sorted_dices[i] = new_dice
                wandb.log({f"best{i}_min_dice": new_dice}, commit=False, step=step)
                with open(f"{wandb.run.dir}/best{i}_w.pkl", "wb") as f:
                    pickle.dump(w, f)

        dice_txt = ",".join(f"{100 * d:02.0f}" for d in dice)
        best_dice_txt = ",".join(f"{100 * d:02.0f}" for d in state.best_sorted_dices)

        t4 = time.perf_counter()

        print(
            (
                f"{wandb.run.dir.split('/')[-2]} "
                f"[{step:04d}:{format_time(time.perf_counter() - state.time0)}] "
                f"test[ dice={dice_txt} "
                f"best_sorted_dices={best_dice_txt}] "
                f"time[ "
                f"E{format_time(t4 - t2)} ]"
            ),
            flush=True,
        )

        epoch_avg_confusion = np.mean(c, axis=0)
        epoch_avg_confusion = epoch_avg_confusion / np.sum(epoch_avg_confusion)

        wandb.log(
            {
                "true_negatives": epoch_avg_confusion[0, 0],
                "true_positives": epoch_avg_confusion[1, 1],
                "false_negatives": epoch_avg_confusion[1, 0],
                "false_positives": epoch_avg_confusion[0, 1],
                "confusion_matrices": c,
                "time_eval": t4 - t2,
            },
            commit=False,
            step=step,
        )

        del dice, dice_txt, c, test_pred, epoch_avg_confusion
    else:
        t4 = t2

    wandb.log(
        {
            "_runtime": time.perf_counter() - state.time0,
            "train_loss": train_loss,
            "avg_train_loss": np.mean(state.losses),
            "min_pred": min_median_max[0],
            "median_pred": min_median_max[1],
            "max_pred": min_median_max[2],
            "time_update": t2 - t1,
        },
        commit=True,
        step=step,
    )

    state = TrainState(
        time0=state.time0,
        train_set=state.train_set,
        test_set=state.test_set,
        t4=t4,
        losses=state.losses,
        best_sorted_dices=state.best_sorted_dices,
    )
    return (state, w, opt_state)


def patch_slices(total: int, size: int, pad: int, overlap: float) -> List[int]:
    r"""
    Generate a list of patch indices such that the center of the patches (unpaded patches) cover the full image.

    Args:
        total: The total size of the image.
        size: The size of the patch.
        pad: The padding of the patch.
        overlap: The overlap of the patches.
    """
    step = max(1, round((size - 2 * pad) / overlap))
    naive = list(range(0, total - size, step)) + [total - size]
    return np.round(np.linspace(0, total - size, len(naive))).astype(int)


def eval_model(
    img: jnp.ndarray,
    apply: Callable[[jnp.ndarray], jnp.ndarray],
    overlap: float,
    size: Tuple[int, int, int] = None,
    padding: Tuple[int, int, int] = None,
    verbose: bool = False,
) -> np.ndarray:
    assert img.ndim == 3 + 1

    if size is None:
        size = sample_size
    if padding is None:
        padding = sample_padding

    pos = np.stack(
        np.meshgrid(
            np.linspace(-1.3, 1.3, size[0] - 2 * padding[0]),
            np.linspace(-1.3, 1.3, size[1] - 2 * padding[1]),
            np.linspace(-1.3, 1.3, size[2] - 2 * padding[2]),
            indexing="ij",
        ),
        axis=-1,
    )
    gaussian = np.exp(-np.linalg.norm(pos, axis=-1) ** 2)

    sum = np.zeros_like(img[:, :, :, 0])
    num = np.zeros_like(img[:, :, :, 0])

    for i in patch_slices(img.shape[0], size[0], padding[0], overlap):
        for j in patch_slices(img.shape[1], size[1], padding[1], overlap):
            for k in patch_slices(img.shape[2], size[2], padding[2], overlap):
                x = img[i : i + size[0], j : j + size[1], k : k + size[2]]
                p = apply(x)
                p = unpad(p, padding)

                sum[
                    i + padding[0] : i + size[0] - padding[0],
                    j + padding[1] : j + size[1] - padding[1],
                    k + padding[2] : k + size[2] - padding[2],
                ] += (
                    p * gaussian
                )
                num[
                    i + padding[0] : i + size[0] - padding[0],
                    j + padding[1] : j + size[1] - padding[1],
                    k + padding[2] : k + size[2] - padding[2],
                ] += gaussian

                if verbose:
                    print(i, j, k, flush=True)

    negative_value = -10.0
    sum[num == 0] = negative_value
    num[num == 0] = 1.0

    return sum / num
