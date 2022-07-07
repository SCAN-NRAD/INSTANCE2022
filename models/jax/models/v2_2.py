from typing import Tuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import numpy as np
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom
import jax


Zooms = Tuple[float, float, float]


def print_stats(name, x):
    return
    assert isinstance(x, e3nn.IrrepsData)
    print(
        name,
        x.shape,
        " ".join(
            f"{mulir}:" + (f"{jnp.mean(a):.1f}+{jnp.mean(a**2)**0.5:.2f}" if a is not None else "none")
            for mulir, a in zip(x.irreps, x.list)
        ),
        flush=True,
    )


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

        def irreps(m):
            return f"{4 * m}x0e + {4 * m}x0o + {2 * m}x1e + {2 * m}x1o + {1 * m}x2e + {1 * m}x2o"

        def instance_norm(x):
            return e3nn.BatchNorm(eps=config.instance_norm_eps, affine=False, instance=True)(x)

        def conv(x, zoom, diameter, irreps):
            x = Convolution(
                irreps,
                diameter=diameter,
                steps=zoom,
                irreps_sh=e3nn.Irreps.spherical_harmonics(lmax=4, p=-1),
                num_radial_basis={0: 4, 1: 4, 2: 3, 3: 2, 4: 1},
                relative_starts={0: 0.0, 1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75},
            )(x)
            x = instance_norm(x)
            return x

        def poly_act(x, m):
            x = e3nn.TensorSquare(irreps(m))(x)
            x = instance_norm(x)
            x = e3nn.TensorSquare(f"{16 * m}x0e")(x)
            x = instance_norm(x)
            x = e3nn.scalar_activation(x, [jax.nn.gelu])
            return x

        m = config.width
        x1 = e3nn.IrrepsData.from_contiguous("3x0e", input)
        z1 = zooms
        del input, zooms
        x1 = x1[None]  # add a batch index
        print_stats("input", x1)

        x1 = conv(x1, z1, 0.37 * 9.0, f"{4 * m}x0e + {2 * m}x1o + {1 * m}x2e")
        x1 = poly_act(x1, m)

        x2, z2 = downsample((x1, z1), 0.8)

        x2 = conv(x2, z2, 0.8 * 9.0, irreps(m))
        x2 = poly_act(x2, 2 * m)

        x3, z3 = downsample((x2, z2), 1.6)

        x3 = conv(x3, z3, 1.6 * 9.0, irreps(2 * m))
        x3 = poly_act(x3, 4 * m)

        x2, _ = upsample((x2, z2), (x3, z3))

        x2 = conv(x2, z2, 0.8 * 9.0, irreps(m))
        x2 = poly_act(x2, 2 * m)

        x1, _ = upsample((x1, z1), (x2, z2))

        x1 = conv(x1, z1, 0.37 * 9.0, irreps(m))
        x1 = e3nn.TensorSquare(irreps(m))(x1)
        x1 = instance_norm(x1)
        x1 = e3nn.TensorSquare("1x0e")(x1)

        return x1[0].contiguous[..., 0]

    return model
