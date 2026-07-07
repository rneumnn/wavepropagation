from ..core.core_classes import RayBundle, Surface
import numpy as np

def propagate_to_surface(rays: RayBundle, surface: Surface):
    t, hit = surface.intersect(rays)
    out = rays.copy()
    valid = out.valid & hit & np.isfinite(t)
    out = out.translate(t)
    out.positions[~valid] = rays.positions[~valid]
    out.valid &= valid
    out.surface = surface
    out.action ="propergate"

    return out