import math
import pickle
import random
from typing import Tuple

import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import time
import wandb
from collections import namedtuple


def load_miccai22(path: str, i: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """i goes from 1 to 100"""
    image = nib.load(f"{path}/data/{i:03d}.nii.gz")
    label = nib.load(f"{path}/label/{i:03d}.nii.gz")
    assert np.allclose(image.affine, label.affine)

    zooms = image.header.get_zooms()
    image = image.get_fdata()
    label = label.get_fdata()

    image = 2.5 * np.clip(image / 80.0, a_min=0.0, a_max=1.0)
    image = image.astype(np.float32)

    label = 2.0 * label - 1.0
    label = label.astype(np.float32)
    return image, label, zooms


def random_slice(size: int, target_size: int) -> slice:
    if size <= target_size:
        return slice(None)
    start = random.randint(0, size - target_size)
    return slice(start, start + target_size)


def random_sample(x: np.ndarray, y: np.ndarray, target_sizes: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    slices = jax.tree_map(random_slice, x.shape, target_sizes)
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


TrainState = namedtuple("TrainState", ["time0", "sample_size", "train_set", "test_set", "confusion_matrices", "t4", "losses"])


def init_train_loop(args, w, opt_state) -> TrainState:
    print("Prepare for the training loop...", flush=True)
    time0 = time.perf_counter()
    sample_size = (100, 100, 25)  # physical size ~= 50mm x 50mm x 125mm

    train_set = [load_miccai22(args.data, i) for i in range(1, 90 + 1)]

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

    confusion_matrices = np.zeros((len(test_set), 2, 2))
    t4 = time.perf_counter()
    losses = np.ones((len(train_set),))

    return TrainState(
        time0=time0,
        sample_size=sample_size,
        train_set=train_set,
        test_set=test_set,
        confusion_matrices=confusion_matrices,
        t4=t4,
        losses=losses,
    )


def train_loop(args, state: TrainState, i, w, opt_state, un, update, apply_model) -> TrainState:
    t0 = time.perf_counter()
    t_extra = t0 - state.t4

    if i == 120:
        jax.profiler.start_trace(wandb.run.dir)

    img, lab, zooms = state.train_set[i % len(state.train_set)]

    # regroup zooms and sizes by rounding and taking subsets of the volume
    zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
    if np.random.rand() < 0.5:
        while True:
            x, y = random_sample(img, lab, state.sample_size)
            if np.any(un(y) == 1):
                img, lab = x, y
                break
    else:
        img, lab = random_sample(img, lab, state.sample_size)

    t1 = time.perf_counter()

    w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, args.lr)
    train_loss.block_until_ready()

    if args.dummy:
        time.sleep(0.5)

    t2 = time.perf_counter()
    c = state.confusion_matrices
    if i % 3 == 0:
        j = (i // 3) % 10
        img, lab, zooms = state.test_set[j]
        test_pred = apply_model(w, img, zooms)
        c[j] = np.array(confusion_matrix(un(lab), un(test_pred)))

    t3 = time.perf_counter()
    epoch_avg_confusion = np.mean(c, axis=0)
    epoch_avg_confusion = epoch_avg_confusion / np.sum(epoch_avg_confusion)

    with np.errstate(invalid="ignore"):
        dice = 2 * c[:, 1, 1] / (2 * c[:, 1, 1] + c[:, 1, 0] + c[:, 0, 1])

    min_median_max = np.min(train_pred), np.median(train_pred), np.max(train_pred)

    state.losses[i % len(state.train_set)] = train_loss
    t4 = time.perf_counter()

    dice_txt = ",".join(f"{d:.2f}" for d in dice)
    print(
        (
            f"{wandb.run.dir.split('/')[-2]} "
            f"[{i + 1:04d}:{format_time(time.perf_counter() - state.time0)}] "
            f"train[ loss={np.mean(state.losses):.4f} "
            f"mi-me-ma={min_median_max[0]:.1f} {min_median_max[1]:.1f} {min_median_max[2]:.1f} ] "
            f"test[ dice={dice_txt} ] "
            f"time[ "
            f"S{format_time(t1 - t0)}+"
            f"U{format_time(t2 - t1)}+"
            f"E{format_time(t3 - t2)}+"
            f"C{format_time(t4 - t3)}+"
            f"EX{format_time(t_extra)} ]"
        ),
        flush=True,
    )

    wandb_state = {
        "iteration": i,
        "_runtime": time.perf_counter() - state.time0,
        "train_loss": train_loss,
        "true_negatives": epoch_avg_confusion[0, 0],
        "true_positives": epoch_avg_confusion[1, 1],
        "false_negatives": epoch_avg_confusion[1, 0],
        "false_positives": epoch_avg_confusion[0, 1],
        "min_pred": min_median_max[0],
        "median_pred": min_median_max[1],
        "max_pred": min_median_max[2],
        "time_update": t2 - t1,
        "time_eval": t3 - t2,
        "confusion_matrices": c,
    }.update({f"dice_{91 + i}": d for i, d in enumerate(dice)})

    if i % 500 == 0:
        with open(f"{wandb.run.dir}/w.pkl", "wb") as f:
            pickle.dump(w, f)

    if i == 120:
        jax.profiler.stop_trace()

    wandb.log(wandb_state)

    state = TrainState(
        time0=state.time0,
        sample_size=state.sample_size,
        train_set=state.train_set,
        test_set=state.test_set,
        confusion_matrices=c,
        t4=t4,
        losses=state.losses,
    )
    return (state, w, opt_state)
