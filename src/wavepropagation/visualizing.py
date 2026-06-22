#spectrum plotting
#3d multiple fieldplanes plotting
#polarization
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .field import RadialField, Field, FieldBase
from .spectrum import PolychromaticField, is_visible
import numpy as np
#from .utils import is_visible as isV

def _set_labels(field:FieldBase, axes:plt.Axes, mode:str="field"):
    accepted_modes = {"field":("Field", "E", "a.u."),
                      "intensity":("Intensity", "I", "a.u.")}
    if mode not in accepted_modes.keys():
        raise ValueError(f"mode needs to be one of {[x for x in accepted_modes.keys()]}")
    if field.is_radial:
        axes.set_ylabel(f"{accepted_modes[mode][0]} [{accepted_modes[mode][2]}]")
    else:
        axes.set_ylabel("y [mm]")
    axes.set_xlabel("x [mm]")
    if field.last_element is not None:
        axes.set_title(field.last_element.name)
    else:
        axes.set_title("Initial field")

def axes_dim(figure:Figure, margin):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 4 * margin) / w,
                      (h - 2 * margin) / h]

### radial plotting
def plot_radial_field_Ex(field:RadialField, color=None, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, minWL = 380, maxWL = 780, cmap = "turbo")->Figure|None:
    
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l, margin))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig, margin))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()
    
    if color is None:
        if is_visible(field.wavelength):
            color = PolychromaticField.wavelength_to_rgb(field.wavelength*1e9)
        else:
            color = PolychromaticField.wavelength_to_falsecolor(
                field.wavelength*1e9,
                wavelength_max_nm=maxWL,
                wavelength_min_nm=minWL,
                cmap=cmap)

    axes.plot(field.grid.r, weight*field.Ex, color = color, linestyle = "-", label = f"{field.wavelength*1e9:.2f} nm")
    _set_labels(field, axes, "field")
    if fig is None: return fig_l

def plot_radial_field_Ey(field:RadialField, color=None, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, minWL = 380, maxWL = 780, cmap = "turbo")->Figure|None:

    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l, margin))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig, margin))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()
    
    if color is None:
        if is_visible(field.wavelength):
            color = PolychromaticField.wavelength_to_rgb(field.wavelength*1e9)
        else:
            color = PolychromaticField.wavelength_to_falsecolor(
                field.wavelength*1e9,
                wavelength_max_nm=maxWL,
                wavelength_min_nm=minWL,
                cmap=cmap)

    axes.plot(field.grid.r, weight*field.Ey, color = color, linestyle = "-", label = f"{field.wavelength*1e9:.2f} nm")
    _set_labels(field, axes, "field")
    if fig is None: return fig_l


def plot_radial_intensity(field:RadialField, color=None, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, minWL = 380, maxWL = 780, cmap = "turbo")->Figure|None:

    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l, margin))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig, margin))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()

    if color is None:
        if is_visible(field.wavelength):
            color = PolychromaticField.wavelength_to_rgb(field.wavelength*1e9)
        else:
            color = PolychromaticField.wavelength_to_falsecolor(
                field.wavelength*1e9,
                wavelength_max_nm=maxWL,
                wavelength_min_nm=minWL,
                cmap=cmap)

    axes.plot(field.grid.r, weight*field.intensity(), color = color, linestyle = "-", label = f"{field.wavelength*1e9:.2f} nm")
    _set_labels(field,axes,"intensity")
    if fig is None: return fig_l

def plot_polychromaticField_Ex(polyField:PolychromaticField,cmap = "turbo", fig:Figure|None = None, axes:plt.Axes = None)->Figure|None:
    fig_l = None
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
        else:
            fig_l = fig
    else: fig = axes.get_figure()

    if polyField.is_radial:
        for comp in polyField.components:
            plot_radial_field_Ex(comp.field, weight = comp.weight, fig=fig_l, axes = axes,
                                 minWL=polyField.wavelengths.min()*1e9,
                                 maxWL=polyField.wavelengths.max()*1e9,
                                 cmap=cmap)
        fig_l.legend()
        if fig is None:
            return fig_l
    else: raise(NotImplementedError)

def plot_polychromaticField_intensity(polyField:PolychromaticField,cmap = "turbo", fig:Figure|None = None, axes:plt.Axes = None)->Figure|None:
    fig_l = None
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
        else:
            fig_l = fig
    else: fig = axes.get_figure()

    if polyField.is_radial:
        for comp in polyField.components:
            plot_radial_intensity(comp.field, weight = comp.weight, fig=fig_l, axes=axes,
                                 minWL=polyField.wavelengths.min()*1e9,
                                 maxWL=polyField.wavelengths.max()*1e9,
                                 cmap=cmap)
        fig_l.legend()
        if fig is None:
            return fig_l
    else: raise(NotImplementedError)



### 2d plotting
def plot_field2d_Ex(field:Field, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, cmap = "plasma", colorbar:bool = False)->Figure|None:
    def axes_dim(figure:Figure):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 2 * margin) / w,
                    (h - 2 * margin) / h]
    extents = np.ndarray(-field.grid.L/2, field.grid.L/2, -field.grid.L/2, field.grid.L/2)*1e3
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()

    img = axes.imshow(field.Ex.real, cmap=cmap, extent=extents)
    _set_labels(field,axes,"field")
    if colorbar: plt.colorbar(img)
    
    if fig is None: return fig_l

def plot_field2d_Ey(field:Field, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, cmap = "plasma", colorbar:bool = False)->Figure|None:
    def axes_dim(figure:Figure):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 2 * margin) / w,
                    (h - 2 * margin) / h]
    extents = np.ndarray(-field.grid.L/2, field.grid.L/2, -field.grid.L/2, field.grid.L/2)*1e3
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()

    img = axes.imshow(field.Ey.real, cmap=cmap, extent=extents)
    _set_labels(field,axes, "field")
    if colorbar: plt.colorbar(img)
    
    if fig is None: return fig_l
    
def plot_field2d_Intensity(field:Field, weight = 1, fig:Figure = None, axes:plt.Axes = None, margin = .3, cmap = "plasma", colorbar:bool = False)->Figure|None:
    def axes_dim(figure:Figure):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 2 * margin) / w,
                    (h - 2 * margin) / h]
    extents = np.array((-field.grid.L/2, field.grid.L/2, -field.grid.L/2, field.grid.L/2))*1e3
    if (axes is not None) and (fig is not None):
        print(f"WARNING in {__name__}: Both 'fig' and 'axes' are given. 'Axes' will always be prioritized!")
    if axes is None:
        if fig is None:
            fig_l = plt.figure()
            axes = fig_l.add_axes(axes_dim(fig_l))
        else:
            axes = fig.get_axes()
            if len(axes) == 0:
                axes = fig.add_axes(axes_dim(fig))
            else:
                axes = axes[0]
    else: fig = axes.get_figure()

    img = axes.imshow(field.intensity(), cmap=cmap, extent=extents)
    _set_labels(field,axes, "intensity")
    if colorbar: plt.colorbar(img)
    
    if fig is None: return fig_l
