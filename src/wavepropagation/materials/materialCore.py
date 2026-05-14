# material.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


RefractiveIndexFunction = Callable[[float | np.ndarray], float | np.ndarray]


def sellmeier_n(B, C) -> RefractiveIndexFunction:
    """Returns a function that calculates the refractive index n(wavelength) using the Sellmeier equation with coefficients B and C.
    The Sellmeier equation is given by:
        n^2(wavelength) = 1 + sum(B_i * wavelength^2 / (wavelength^2 - C_i))
        where B_i and C_i are the Sellmeier coefficients in meters and meters^2, respectively.
        :param B: list or array of Sellmeier coefficients B_i
        :param C: list or array of Sellmeier coefficients C_i
        :return: function n(wavelength) that calculates the refractive index for a given wavelength or array of wavelengths
    """
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    if B.shape != C.shape:
        raise ValueError("B and C must have the same shape.")

    def n(wavelength):
        wl = np.asarray(wavelength, dtype=float)
        wl2 = wl**2
        terms = B * wl2[..., None] / (wl2[..., None] - C)
        return np.sqrt(1.0 + np.sum(terms, axis=-1))

    return n


@dataclass(frozen=True)
class Material:
    name: str
    n_function: RefractiveIndexFunction
    sellmeier_coefficients: tuple[np.ndarray, np.ndarray] | None = None
    lambda_d: float | None = 587.56e-9 # nm
    lambda_C: float | None = 656.3e-9 # nm
    lambda_F: float | None = 486.1e-9 # nm

    def n(self, wavelength: float | np.ndarray):
        return self.n_function(wavelength)

    @classmethod
    def constant(cls, n: float, name: str = "constant material") -> "Material":
        def n_function(wavelength):
            wl = np.asarray(wavelength)
            return np.full_like(wl, fill_value=n, dtype=float)

        return cls(name=name, n_function=n_function)

    @classmethod
    def sellmeier(
        cls,
        name: str,
        B: list[float],
        C: list[float],
    ) -> "Material":
        B_arr = np.asarray(B, dtype=float)
        C_arr = np.asarray(C, dtype=float)

        return cls(
            name=name,
            n_function=sellmeier_n(B_arr, C_arr),
            sellmeier_coefficients=(B_arr, C_arr),
        )
    

    def v_number(self, center_wl: float, wl_short: float, wl_long: float)-> float:
        return (self.n(center_wl)-1) / (self.n(wl_short)-self.n(wl_long))
    
    def abbe_number(self) -> float:
        return (self.n(self.lambda_d) - 1)/(self.n(self.lambda_F)-self.n(self.lambda_C))