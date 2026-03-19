from dataclasses import dataclass
import numpy as np
from .field import Field


@dataclass
class SpectralComponent:
    wavelength: float
    weight: float
    field: Field


class PolychromaticField:
    def __init__(self, components: list[SpectralComponent]):
        if not components:
            raise ValueError("components must not be empty")

        grid = components[0].field.grid
        for c in components:
            if c.field.grid is not grid:
                raise ValueError("All components must share the same Grid instance.")
            if abs(c.field.wavelength - c.wavelength) > 0:
                raise ValueError("Component wavelength and field wavelength must match.")

        self.grid = grid
        self.components = list(components)

    def copy(self) -> "PolychromaticField":
        return PolychromaticField([
            SpectralComponent(
                wavelength=c.wavelength,
                weight=c.weight,
                field=c.field.copy(),
            )
            for c in self.components
        ])

    def wavelengths(self) -> np.ndarray:
        return np.array([c.wavelength for c in self.components], dtype=float)

    def weights(self) -> np.ndarray:
        return np.array([c.weight for c in self.components], dtype=float)

    def intensity(self) -> np.ndarray:
        total = np.zeros((self.grid.N, self.grid.N), dtype=float)
        for c in self.components:
            total += c.weight * c.field.intensity()
        return total

    def total_power(self) -> float:
        return float(sum(c.weight * c.field.power() for c in self.components))
    
    #visualization helper method to get RGB color for each component based on wavelength
    def rgb_image(self, gamma: float = 1.0, normalize: bool = True) -> np.ndarray:
        img = np.zeros((self.grid.N, self.grid.N, 3), dtype=float)

        for c in self.components:
            rgb = wavelength_to_rgb(c.wavelength * 1e9)
            img += (c.weight * c.field.intensity())[..., None] * rgb[None, None, :]

        if normalize:
            max_val = img.max()
            if max_val > 0:
                img /= max_val

        if gamma != 1.0:
            img = np.clip(img, 0.0, 1.0) ** (1.0 / gamma)

        return np.clip(img, 0.0, 1.0)
    

def wavelength_to_rgb(wavelength_nm: float) -> np.ndarray:
    wl = float(wavelength_nm)

    if wl < 380 or wl > 780:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 420 <= wl < 701:
        factor = 1.0
    else:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)

    return np.clip(np.array([r, g, b], dtype=float) * factor, 0.0, 1.0)

