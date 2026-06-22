from .field import FieldBase
from .spectrum import PolychromaticField, SpectralComponent
from .elements import element_base
import numpy as np


class OpticalSystem:
    def __init__(self, elements: list[element_base]):
        self.elements = list(elements)

    def run(self, obj: FieldBase|PolychromaticField, **kwargs) -> tuple[FieldBase|PolychromaticField, list[FieldBase|PolychromaticField]|None]:
        keep_history = kwargs.get('keep_history', False)

        def apply_element(element:element_base, field:FieldBase|PolychromaticField) -> FieldBase|PolychromaticField:
            current = element.apply(field)
            if keep_history:
                    historyField = current.copy()
                    return current, historyField
            return current, None

        if isinstance(obj, FieldBase):
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
                    print(f"Applying element {elem.name} to component {c} with wavelength {comp.wavelength*1e9:.2f} nm")
                    current, hist = apply_element(elem, current)
                    if keep_history:
                        history_components[c, e] = (
                            SpectralComponent(
                                wavelength=comp.wavelength,
                                weight=comp.weight,
                                omega = comp.omega,
                                field=hist,
                                sampling_method=comp.sampling_method
                            )
                        )
                out_components.append(
                    SpectralComponent(
                        wavelength=comp.wavelength,
                        weight=comp.weight,
                        omega = comp.omega,
                        field=current,
                        sampling_method=comp.sampling_method
                    )
                )
            if keep_history:
                #fetch history for each element toget a list of PolychromaticField for each element
                history = []
                for e in range(len(self.elements)):
                    history.append(PolychromaticField(history_components[:, e]))
                    #set last element
                return PolychromaticField(out_components), history
            return PolychromaticField(out_components), None

        raise TypeError("Unsupported object type")
    

    class HistoryControl:
        """
        Class to controll the history treatment. Should be used to do complex evaluation/measurement methods.
        Todo: integrate to field, refine the concept
        """
        def __init__(self):
            self.history_index = [0]
            self.history_name = ['source']
            return
        
        def add_history_name(self, name:str|list[str]):
            name = list(name)
            self.history_name.extend(name)

        def add_history_index(self, index:int|list[int]):
            index = list(index)
            self.history_name.extend(index)