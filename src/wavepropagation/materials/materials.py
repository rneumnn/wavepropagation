import numpy as np
from .materialCore import Material

# Fused silica, Malitson coefficients
# wavelength in meters, so C is in m^2

FUSED_SILICA = Material.sellmeier(
    name="Fused Silica",
    B=[
        0.6961663,
        0.4079426,
        0.8974794,
    ],
    C=[
        (0.0684043e-6) ** 2,
        (0.1162414e-6) ** 2,
        (9.896161e-6) ** 2,
    ],
)

AIR = Material.constant(1.0, name="Air")

BK7 = Material.sellmeier(
    name="N-BK7",
    B=[
        1.03961212,
        0.231792344,
        1.01046945,
    ],
    C=[
        (0.00600069867) * 1e-12,
        (0.0200179144) * 1e-12,
        (103.560653) * 1e-12,
    ],
)

