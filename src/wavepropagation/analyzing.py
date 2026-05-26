#different analysation tools for the field, e.g. spectrum, polarization, beam quality, etc.
# stokes
# routines that can be passed directly to the optical system definititon for exemle analyzing polarization

import numpy as np


def phase_sampling_requirement(phase: np.ndarray, safety: float = 1.0):
    """
    Estimate whether a phase mask is sufficiently sampled.

    safety:
        Desired maximum phase step per pixel.
        Use 1.0 for good sampling.
    """
    dphi_x = np.nanmax(np.abs(np.diff(phase, axis=1)))
    dphi_y = np.nanmax(np.abs(np.diff(phase, axis=0)))
    dphi_max = max(dphi_x, dphi_y)

    factor = dphi_max / safety

    print(f"max phase step: {dphi_max:.3f} rad")
    print(f"target step:    {safety:.3f} rad")

    if factor <= 1:
        print("OK")
    else:
        print(f"Need roughly {factor:.1f}x more samples per dimension.")

    return factor

def check_phase_sampling(phase: np.ndarray, name: str = "phase", safety: float = 1.0):
    """
    Checks whether a 2D phase mask is spatially well sampled.

    phase:
        unwrapped phase in rad. Based on the grid of the field, the phase should be sampled at least every pi rad to avoid aliasing artifacts. This function computes the maximum phase difference between adjacent pixels and compares it to pi.
    """
    phase = np.asarray(phase, dtype=float)

    dphi_x = np.abs(np.diff(phase, axis=1))
    dphi_y = np.abs(np.diff(phase, axis=0))

    max_dphi_x = np.nanmax(dphi_x)
    max_dphi_y = np.nanmax(dphi_y)
    max_dphi = max(max_dphi_x, max_dphi_y)

    print(f"{name}:")
    print(f"  max |dphi/dx pixel| = {max_dphi_x:.3f} rad")
    print(f"  max |dphi/dy pixel| = {max_dphi_y:.3f} rad")
    print(f"  max |dphi|          = {max_dphi:.3f} rad")
    print(f"  pi                  = {np.pi:.3f} rad")

    if max_dphi > np.pi:
        print("  BAD: phase is undersampled.")
    elif max_dphi > 1.0:
        print("  BORDERLINE: phase may show artifacts.")
    else:
        print("  OK: phase sampling looks good.")

    sampling_required_factor = phase_sampling_requirement(phase, safety=safety)

    return max_dphi, sampling_required_factor

import numpy as np


def required_N_for_lens_phase(
    wavelength: float,
    n_medium: float,
    L: float,
    f: float,
    r_max: float,
    max_phase_step: float = 1.0,
) -> int:
    """
    Estimate required grid size N for a thin lens phase mask.

    Parameters
    ----------
    wavelength:
        Vacuum wavelength [m].

    n_medium:
        Refractive index of propagation medium.

    L:
        Grid side length [m].

    f:
        Focal length [m].

    r_max:
        Maximum radius where the field is relevant [m].

    max_phase_step:
        Maximum allowed phase step per pixel [rad].
        Use 1.0 for safe sampling, np.pi for Nyquist-like limit.

    Returns
    -------
    N_required:
        Estimated minimum grid size.
    """
    k = 2 * np.pi * n_medium / wavelength

    N_required = k * r_max * L / (f * max_phase_step)

    return int(np.ceil(N_required))