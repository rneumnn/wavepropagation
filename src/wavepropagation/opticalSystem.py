from .field import Field
from .spectrum import PolychromaticField, SpectralComponent


class OpticalSystem:
    def __init__(self, elements):
        self.elements = list(elements)

    def run(self, obj):
        if isinstance(obj, Field):
            current = obj.copy()
            for elem in self.elements:
                current = elem.apply(current)
            return current

        if isinstance(obj, PolychromaticField):
            out_components = []
            for comp in obj.components:
                current = comp.field.copy()
                for elem in self.elements:
                    current = elem.apply(current)
                out_components.append(
                    SpectralComponent(
                        wavelength=comp.wavelength,
                        weight=comp.weight,
                        field=current,
                    )
                )
            return PolychromaticField(out_components)

        raise TypeError("Unsupported object type")