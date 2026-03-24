from .field import Field
from .spectrum import PolychromaticField, SpectralComponent
from .elements import element_base
import numpy as np


class OpticalSystem:
    def __init__(self, elements: list[element_base]):
        self.elements = list(elements)

    def run(self, obj: Field|PolychromaticField, **kwargs) -> tuple[Field|PolychromaticField, list[Field|PolychromaticField]|None]:
        keep_history = kwargs.get('keep_history', False)

        def apply_element(element:element_base, field:Field|PolychromaticField) -> Field|PolychromaticField:
            current = element.apply(field)
            if keep_history:
                    historyField = current.copy()
                    return current, historyField
            return current, None

        if isinstance(obj, Field):
            if keep_history:
                history = [obj.copy()]
            current = obj.copy()
            for elem in self.elements:
                current, hist = apply_element(elem, current)
                if keep_history: history.append(hist)
            if keep_history:
                return current, history
            return current, None

        if isinstance(obj, PolychromaticField):
            out_components = []
            if keep_history:
                history_components = np.empty((len(obj.components), len(self.elements)), dtype=object)
            for c,comp in enumerate(obj.components):
                current = comp.field.copy()
                for e, elem in enumerate(self.elements):
                    current, hist = apply_element(elem, current)
                    if keep_history:
                        history_components[c, e] = (
                            SpectralComponent(
                                wavelength=comp.wavelength,
                                weight=comp.weight,
                                field=hist
                            )
                        )
                out_components.append(
                    SpectralComponent(
                        wavelength=comp.wavelength,
                        weight=comp.weight,
                        field=current,
                    )
                )
            if keep_history:
                #fetch history for each element toget a list of PolychromaticField for each element
                history = []
                for e in range(len(self.elements)):
                    history.append(PolychromaticField(history_components[:, e]))
                return PolychromaticField(out_components), history
            return PolychromaticField(out_components), None

        raise TypeError("Unsupported object type")