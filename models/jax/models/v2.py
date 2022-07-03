from typing import Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom


def create_model(config):
    def model(input: jnp.ndarray, zooms: Tuple[float, float, float]) -> jnp.ndarray:
        r"""Rotations, Translations and Mirror Equivariant Unet

        Args:
            input (jnp.ndarray): input data of shape ``(x, y, z, channels)``
            zooms (Tuple[float, float, float]): the zooms of the input data

        Returns:
            jnp.ndarray: output data of shape ``(x, y, z)``
        """
        return input

    return model
