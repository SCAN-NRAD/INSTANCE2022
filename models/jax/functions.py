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
    ["time0", "sample_size", "train_set", "test_set", "confusion_matrices", "t4", "losses", "best_w", "best_min_dice"],
)


def init_train_loop(args, old_state, step, w, opt_state) -> TrainState:
    print("Prepare for the training loop...", flush=True)
    sample_size = (100, 100, 25)  # physical size ~= 50mm x 50mm x 125mm

    if args.dummy:
        train_idx = [1, 2]
        test_idx = [91, 92]
    else:
        train_idx = list(range(args.trainset_start, args.trainset_stop + 1))
        test_idx = [i for i in list(range(1, 100 + 1)) if i not in train_idx]

    train_set = [load_miccai22(args.data, i) for i in train_idx]

    test_set = []
    for i in test_idx:
        img, lab, zooms = load_miccai22(args.data, i)  # test data
        zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
        center_of_mass = np.stack(np.nonzero(lab == 1.0), axis=-1).mean(0).astype(np.int)
        start = np.maximum(center_of_mass - np.array(sample_size) // 2, 0)
        end = np.minimum(start + np.array(sample_size), np.array(img.shape[:3]))
        start = end - np.array(sample_size)
        img = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        lab = lab[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        test_set.append((img, lab, zooms))

    return TrainState(
        time0=getattr(old_state, "time0", time.perf_counter()),
        sample_size=sample_size,
        train_set=train_set,
        test_set=test_set,
        confusion_matrices=np.zeros((len(test_set), 2, 2)),
        t4=time.perf_counter(),
        losses=np.ones((len(train_set),)),
        best_w=getattr(old_state, "best_w", jax.tree_map(np.array, w)),
        best_min_dice=getattr(old_state, "best_min_dice", 0.0),
    )


def train_loop(args, state: TrainState, step, w, opt_state, un, update, apply_model) -> TrainState:
    t0 = time.perf_counter()
    t_extra = t0 - state.t4

    if step == 120:
        jax.profiler.start_trace(wandb.run.dir)

    img, lab, zooms = state.train_set[step % len(state.train_set)]

    # regroup zooms and sizes by rounding and taking subsets of the volume
    zooms = jax.tree_map(lambda x: round(433 * x) / 433, zooms)
    if np.random.rand() < 0.5:
        # avoid patch without label
        while True:
            x, y = random_sample(img, lab, state.sample_size)
            if np.any(un(y) == 1):
                img, lab = x, y
                break
    else:
        # avoid patch full of air
        while True:
            x, y = random_sample(img, lab, state.sample_size)
            if np.any(x > 0.0):
                img, lab = x, y
                break
    del x, y

    t1 = time.perf_counter()

    lr = args.lr * max(0.1 ** math.floor(step / args.lr_div_step), 0.01)
    w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, lr)
    train_loss.block_until_ready()

    t2 = time.perf_counter()
    c = np.array(confusion_matrix(un(lab), un(train_pred)))
    with np.errstate(invalid="ignore"):
        train_dice = 2 * c[1, 1] / (2 * c[1, 1] + c[1, 0] + c[0, 1])

    min_median_max = np.min(train_pred), np.median(train_pred), np.max(train_pred)

    state.losses[step % len(state.train_set)] = train_loss

    print(
        (
            f"{wandb.run.dir.split('/')[-2]} "
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

    best_min_dice = state.best_min_dice
    best_w = state.best_w

    if step % 50 == 0:
        c = state.confusion_matrices
        for j, (img, lab, zooms) in enumerate(state.test_set):
            test_pred = apply_model(w, img, zooms)
            c[j] = np.array(confusion_matrix(un(lab), un(test_pred)))

        with np.errstate(invalid="ignore"):
            dice = 2 * c[:, 1, 1] / (2 * c[:, 1, 1] + c[:, 1, 0] + c[:, 0, 1])

        wandb.log({f"dice_{91 + i}": d for i, d in enumerate(dice)}, commit=False, step=step)

        if np.min(dice) > state.best_min_dice:
            best_min_dice = np.min(dice)
            best_w = jax.tree_map(np.array, w)
            wandb.log({"best_min_dice": best_min_dice}, commit=False, step=step)

            with open(f"{wandb.run.dir}/best_w.pkl", "wb") as f:
                pickle.dump(best_w, f)

        dice_txt = ",".join(f"{100 * d:02.0f}" for d in dice)

        t4 = time.perf_counter()

        print(
            (
                f"{wandb.run.dir.split('/')[-2]} "
                f"[{step:04d}:{format_time(time.perf_counter() - state.time0)}] "
                f"test[ dice={dice_txt} best_min_dice={100 * best_min_dice:02.0f}] "
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
        sample_size=state.sample_size,
        train_set=state.train_set,
        test_set=state.test_set,
        confusion_matrices=state.confusion_matrices,
        t4=t4,
        losses=state.losses,
        best_w=best_w,
        best_min_dice=best_min_dice,
    )
    return (state, w, opt_state)
