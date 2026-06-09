#spectrum plotting
#3d multiple fieldplanes plotting
#polarization
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .field import RadialField
from .spectrum import PolychromaticField, is_visible
#from .utils import is_visible as isV

def plot_radial_field_Ex(field:RadialField, color=None, weight = 1, fig:Figure = None, margin = .1, minWL = 380, maxWL = 780, cmap = "turbo")->Figure|None:
    def axes_dim(figure:Figure):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 2 * margin) / w,
                      (h - 2 * margin) / h]
    if fig is None:
        fig_l = plt.figure()
        axes = fig_l.add_axes(axes_dim(fig_l))
    else:
        axes = fig.get_axes()
        if len(axes) == 0:
            axes = fig.add_axes(axes_dim(fig))
        else:
            axes = axes[0]
    if color is None:
        if is_visible(field.wavelength):
            color = PolychromaticField.wavelength_to_rgb(field.wavelength*1e9)
        else:
            color = PolychromaticField.wavelength_to_falsecolor(
                field.wavelength*1e9,
                wavelength_max_nm=maxWL,
                wavelength_min_nm=minWL,
                cmap=cmap)

    axes.plot(field.grid.r, weight*field.Ex, color = color, linestyle = "-", label = f"{field.wavelength:.2f} nm")
    if fig is None: return fig_l

def plot_radial_intensity(field:RadialField, color=None, weight = 1, fig:Figure = None, margin = .3, minWL = 380, maxWL = 780, cmap = "turbo")->Figure|None:
    def axes_dim(figure:Figure):
        w, h = figure.get_size_inches()
        return [margin / w, margin / h, (w - 2 * margin) / w,
                      (h - 2 * margin) / h]
    if fig is None:
        fig_l = plt.figure()
        axes = fig_l.add_axes(axes_dim(fig_l))
    else:
        axes = fig.get_axes()
        if len(axes) == 0:
            axes = fig.add_axes(axes_dim(fig))
        else:
            axes = axes[0]
    if color is None:
        if is_visible(field.wavelength):
            color = PolychromaticField.wavelength_to_rgb(field.wavelength*1e9)
        else:
            color = PolychromaticField.wavelength_to_falsecolor(
                field.wavelength*1e9,
                wavelength_max_nm=maxWL,
                wavelength_min_nm=minWL,
                cmap=cmap)

    axes.plot(field.grid.r, weight*field.intensity(), color = color, linestyle = "-", label = f"{field.wavelength:.2f} nm")
    if fig is None: return fig_l

def plot_polychromaticField_Ex(polyField:PolychromaticField,cmap = "turbo", fig:Figure|None = None)->Figure|None:
    if fig is None:
        fig_l = plt.figure()
    else:
        fig_l = fig
    if polyField.is_radial:
        for comp in polyField.components:
            plot_radial_field_Ex(comp.field, weight = comp.weight, fig=fig_l,
                                 minWL=polyField.wavelengths.min()*1e9,
                                 maxWL=polyField.wavelengths.max()*1e9,
                                 cmap=cmap)
        fig_l.legend()
        if fig is None:
            return fig_l
    else: raise(NotImplementedError)

def plot_polychromaticField_intensity(polyField:PolychromaticField,cmap = "turbo", fig:Figure|None = None)->Figure|None:
    if fig is None:
        fig_l = plt.figure()
    else:
        fig_l = fig
    if polyField.is_radial:
        for comp in polyField.components:
            plot_radial_intensity(comp.field, weight = comp.weight, fig=fig_l,
                                 minWL=polyField.wavelengths.min()*1e9,
                                 maxWL=polyField.wavelengths.max()*1e9,
                                 cmap=cmap)
        fig_l.legend()
        if fig is None:
            return fig_l
    else: raise(NotImplementedError)