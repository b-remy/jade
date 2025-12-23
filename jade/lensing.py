import jax
import jax.numpy as jnp

from jade.init import FIELD_MEAN, FIELD_STD
from functools import partial

@partial(jax.vmap, in_axes=(-1,-1), out_axes=(-1))
def ks93inv(kE, kB):
    """Direct inversion of weak-lensing convergence to shear.
    This function provides the inverse of the Kaiser & Squires (1993) mass
    mapping algorithm, namely the shear is recovered from input E-mode and
    B-mode convergence maps.
    Parameters
    ----------
    kE, kB : array_like
        2D input arrays corresponding to the E-mode and B-mode (i.e., real and
        imaginary) components of convergence.
    Returns
    -------
    g1, g2 : tuple of numpy arrays
        Maps of the two components of shear.
    Raises
    ------
    AssertionError
        For input arrays of different sizes.
    See Also
    --------
    ks93
        For the forward operation (shear to convergence).
    """
    # Check consistency of input maps
    assert kE.shape == kB.shape

    # Compute Fourier space grids
    (nx, ny) = kE.shape
    k1, k2 = jnp.meshgrid(jnp.fft.fftfreq(ny), jnp.fft.fftfreq(nx))

    # Compute Fourier transforms of kE and kB
    kEhat = jnp.fft.fft2(kE)
    kBhat = jnp.fft.fft2(kB)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    #k2[0, 0] = 1  # avoid division by 0
    #k2 = jax.ops.index_update(k2, jax.ops.index[0, 0], 1) # avoid division by 0
    k2 = k2.at[0,0].set(1)
    
    g1hat = (p1 * kEhat - p2 * kBhat) / k2
    g2hat = (p2 * kEhat + p1 * kBhat) / k2

    # Transform back to pixel space
    g1 = jnp.fft.ifft2(g1hat).real
    g2 = jnp.fft.ifft2(g2hat).real

    return g1, g2

def Operator(kE, field_mean=FIELD_MEAN, field_std=FIELD_STD):
    """
    Wrapping function for the Kaiser & Squires inversion from convergence to shear.
    Assuming zero B-modes.
    
    :param kE: 2D array of E-mode convergence map. shape (nx, ny, nc)
    :return: 2D array of shear map with shape (2, nx, ny, nc)
    """

    kB = jnp.zeros_like(kE)

    kE = kE * FIELD_STD + FIELD_MEAN
    
    g1, g2 = ks93inv(kE, kB)

    return jnp.stack([g1, g2],0)