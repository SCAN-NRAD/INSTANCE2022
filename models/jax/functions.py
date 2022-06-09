import math
import random
from typing import Tuple

import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


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
