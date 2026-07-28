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
    name="BK7",
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

N_BK7_Sellmeier = {
    "B1": 1.03961212,
    "B2": 0.231792344,
    "B3": 1.01046945,
    "C1": 0.00600069867*1e-12,
    "C2": 0.0200179144*1e-12,
    "C3": 103.560653*1e-12,
}

N_BK7 = Material.sellmeier_from_dict("N-BK7", N_BK7_Sellmeier)

N_SK2_Sellmeier = {
    "B1": 1.28189012, "B2": 0.257738258, "B3": 0.96818604,
    "C1": 0.0072719164*1e-12, "C2": 0.0242823527*1e-12, "C3": 110.377773*1e-12,
}
N_SK2 = Material.sellmeier_from_dict("N-SK2", N_SK2_Sellmeier)

N_SF5_Sellmeier = {
    "B1": 1.52481889, "B2": 0.187085527, "B3": 1.42729015,
    "C1": 0.011254756*1e-12, "C2": 0.0588995392*1e-12, "C3": 129.141675*1e-12,
}
N_SF5 = Material.sellmeier_from_dict("N-SF5", N_SF5_Sellmeier)

H_ZF1 = Material.sellmeier(
    "H-ZF1", B= [1.06294164,0.146910876,1.48922543], C=[108.400406e-12, 0.0591797128e-12, 0.0115933111e-12]
)

H_K9L = Material.sellmeier(
    "H-K9L", B=[0.614555251,0.656775017,1.02699346], C=[0.0145987884e-12, 0.00287769588e-12, 107.653051e-12]
)