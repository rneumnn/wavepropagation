from .backend.core import RayBundle
from .backend.surfaces import Surface

def propagate_to_surface(rays:RayBundle, surface:Surface, n_medium):
    t, valid = surface.intersect(rays)

    out = rays.copy()
    distance = t  # if directions are normalized

    out.positions = out.positions + t[..., None] * out.directions
    out.valid &= valid

    out.opl[valid] += n_medium * distance[valid]
    out.phase[valid] += out.k0 * n_medium * distance[valid]

    return out