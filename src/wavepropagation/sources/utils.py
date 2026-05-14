import numpy as np
from scipy.constants import c

def calculate_kr_from_angle(wavelength: float, axicon_half_angle:float, n_axicon:float=1.6, n_medium:float=1.0) -> tuple[float, float]:
    """
    Calculates k_r for the besselbeam definition based on a given axicon with cone oriented in propagation direction.
    
        :param wavelength: field wavelength in meters
        :type wavelength: float
        :param axicon_half_angle: axicon half angle (cone angle to optical axis) in degrees
        :type axicon_half_angle: float
        :param n_axicon: axicon refrective index Default: 1.6
        :type n_axicon: float
        :param n_medium: medium refrective index Default: 1
        :type n_medium: float
        :return: tuple of (kr, kz) where kr is the transverse wavevector component and kz is the longitudinal wavevector component
        :rtype: tuple[float, float]
    """
    #angle of refracted ray to optical axis
    axicon_half_angle = axicon_half_angle*np.pi/180
    k = 2 * np.pi * n_medium / wavelength
    arg = (n_axicon/n_medium)*np.sin(np.pi/2 - axicon_half_angle)
    if np.abs(arg) >= 1: raise ValueError(f"Arg(arcsin) = {arg} > 1 -- Total reflection occures in axicon, chose different parameter for axicon or use 'kr' argument in Besseslbeam function!")
    print(arg)
    theta = np.arcsin(arg) + axicon_half_angle - np.pi/2
    kr = k * np.sin(theta)
    kz = k * np.cos(theta)
    print(f"k = {k}; k_r = {kr}; k_z = {kz}")
    return kr, kz

