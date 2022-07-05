from typing import Tuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom
import jax


Zooms = Tuple[float, float, float]


def downsample(input: Tuple[e3nn.IrrepsData, Zooms], min_zoom: float) -> Tuple[e3nn.IrrepsData, Zooms]:
    data, zooms = input

    z_in = np.array(zooms)
    s_in = np.array(data.shape[-3:])

    s_out = np.floor(s_in * z_in / np.maximum(z_in, min_zoom)).astype(int)
    z_out = z_in * s_in / s_out
    z_out = tuple(float(z) for z in z_out)
    s_out = tuple(int(s) for s in s_out)

    return (
        e3nn.IrrepsData.from_contiguous(
            data.irreps,
            jax.vmap(lambda x: zoom(x, output_size=s_out, interpolation="nearest"), -1, -1)(data.contiguous),
        ),
        z_out,
    )


def upsample(large: Tuple[e3nn.IrrepsData, Zooms], small: Tuple[e3nn.IrrepsData, Zooms]) -> Tuple[e3nn.IrrepsData, Zooms]:
    x_small, z_small = small
    x_large, z_large = large

    upscaled = e3nn.IrrepsData.from_contiguous(
        x_small.irreps,
        jax.vmap(lambda x: zoom(x, output_size=x_large.shape[1:], interpolation="nearest"), -1, -1)(x_small.contiguous),
    )
    assert np.allclose(z_large, np.array(z_small) * np.array(x_small.shape[1:]) / np.array(upscaled.shape[1:]))
    upscaled = e3nn.IrrepsData.cat([upscaled, x_large])
    return (upscaled, z_large)


def create_model(config):
    def model(input: jnp.ndarray, zooms: Zooms) -> jnp.ndarray:
        r"""Rotations, Translations and Mirror Equivariant Unet

        Args:
            input (jnp.ndarray): input data of shape ``(x, y, z, channels)``
            zooms (Tuple[float, float, float]): the zooms of the input data

        Returns:
            jnp.ndarray: output data of shape ``(x, y, z)``
        """
        kw = dict(
            irreps_sh=e3nn.Irreps.spherical_harmonics(lmax=4, p=-1),
            num_radial_basis={0: 4, 1: 4, 2: 3, 3: 2, 4: 1},
            relative_starts={0: 0.0, 1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75},
        )

        def irreps(m):
            return f"{4 * m}x0e + {4 * m}x0o + {2 * m}x1e + {2 * m}x1o + {1 * m}x2e + {1 * m}x2o"

        m = config.width
        x1 = e3nn.IrrepsData.from_contiguous("3x0e", input)
        z1 = zooms
        del input, zooms
        x1 = x1[None]  # add a batch index

        x1 = Convolution(f"{4 * m}x0e + {2 * m}x1o + {1 * m}x2e", diameter=0.37 * 9.0, steps=z1, **kw)(x1)
        x1 = e3nn.TensorSquare(irreps(m))(x1)
        x1 = e3nn.TensorSquare(f"{8 * m}x0e + {8 * m}x0o")(x1)

        x2, z2 = downsample((x1, z1), 0.8)

        x2 = Convolution(irreps(m), diameter=0.8 * 9.0, steps=z2, **kw)(x2)
        x2 = e3nn.TensorSquare(irreps(2 * m))(x2)
        x2 = e3nn.TensorSquare(f"{16 * m}x0e + {16 * m}x0o")(x2)

        x1, _ = upsample((x1, z1), (x2, z2))

        x1 = Convolution(irreps(m), diameter=0.37 * 9.0, steps=z1, **kw)(x1)
        x1 = e3nn.TensorSquare(irreps(m))(x1)
        x1 = e3nn.TensorSquare("1x0e")(x1)

        return x1[0].contiguous[..., 0]

    return model
