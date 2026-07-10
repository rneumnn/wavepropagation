import numpy as np
from opticsSimulationTools.raytracing import frontend as rt
from matplotlib import pyplot as plt
puple=np.array([-1.000,	      	      	    
 -0.950,  	      	      	    
 -0.900,  	      	      	    
 -0.850,  	      	      	    
 -0.800,  	      	      	    
 -0.750,  	      	      	    
 -0.700,  	      	      	    
 -0.650,  	      	      	    
 -0.600,  	      	      	    
 -0.550,  	      	      	    
 -0.500,  	      	      	    
 -0.450,  	      	      	    
 -0.400,  	      	      	    
 -0.350,  	      	      	    
 -0.300,  	      	      	    
 -0.250,  	      	      	    
 -0.200,  	      	      	    
 -0.150,  	      	      	    
 -0.100,  	      	      	    
 -0.050,  	      	      	    
  0.000,  	      	      	    
  0.050,  	      	      	    
  0.100,  	      	      	    
  0.150,  	      	      	    
  0.200,  	      	      	    
  0.250,  	      	      	    
  0.300,  	      	      	    
  0.350,  	      	      	    
  0.400,  	      	      	    
  0.450,  	      	      	    
  0.500,  	      	      	    
  0.550,  	      	      	    
  0.600,  	      	      	    
  0.650,  	      	      	    
  0.700,  	      	      	    
  0.750,  	      	      	    
  0.800,  	      	      	    
  0.850,  	      	      	    
  0.900,  	      	      	    
  0.950,  	      	      	    
  1.000])      	      	    

wl_800 = np.array(
[-0.068602,
-0.079813,
-0.086859,
-0.090311,
-0.090708,
-0.088559,
-0.084339,
-0.078492,
-0.071430,
-0.063536,
-0.055160,
-0.046621,
-0.038209,
-0.030180,
-0.022763,
-0.016154,
-0.010520,
-0.005997,
-0.002690,
-0.000676,
 0.000000,
-0.000676,
-0.002690,
-0.005997,
-0.010520,
-0.016154,
-0.022763,
-0.030180,
-0.038209,
-0.046621,
-0.055160,
-0.063536,
-0.071430,
-0.078492,
-0.084339,
-0.088559,
-0.090708,
-0.090311,
-0.086859,
-0.079813,
-0.068602])*(np.pi*2)

wl_790=np.array([0.095980,
 0.068432,
 0.045945,
 0.027939,
 0.013865,
 0.003208,
-0.004516,
-0.009759,
-0.012939,
-0.014444,
-0.014631,
-0.013825,
-0.012317,
-0.010372,
-0.008220,
-0.006060,
-0.004063,
-0.002366,
-0.001077,
-0.000273,
 0.000000,
-0.000273,
-0.001077,
-0.002366,
-0.004063,
-0.006060,
-0.008220,
-0.010372,
-0.012317,
-0.013825,
-0.014631,
-0.014444,
-0.012939,
-0.009759,
-0.004516,
 0.003208,
 0.013865,
 0.027939,
 0.045945,
 0.068432,
 0.095980])*(np.pi*2)

wl_810=np.array([-0.224955,
-0.220640,
-0.213012,
-0.202635,
-0.190038,
-0.175721,
-0.160154,
-0.143772,
-0.126982,
-0.110160,
-0.093650,
-0.077768,
-0.062797,
-0.048991,
-0.036574,
-0.025739,
-0.016652,
-0.009445,
-0.004222,
-0.001059,
 0.000000,
-0.001059,
-0.004222,
-0.009445,
-0.016652,
-0.025739,
-0.036574,
-0.048991,
-0.062797,
-0.077768,
-0.093650,
-0.110160,
-0.126982,
-0.143772,
-0.160154,
-0.175721,
-0.190038,
-0.202635,
-0.213012,
-0.220640,
-0.224955])*np.pi*2

# telescope parameter

# Lens 1: R1 = -129.75 mm, R2 = infinity, thickness = 3.0 mm
# Lens2: R1 = infinity, R2 = -519.0 mm, thickness = 5.4 mm
# LENS SEPARATION =mm(756.018) S2 - S1,
# after_lens2_PROPAGATION=mm(50.000),

r = 7.5e-3
N_rays = 50
offset = 1e-3

lens1 = rt.ThickRealLens(
    R1 = -129.75e-3, R2 = 0,
    center_thickness= 3e-3,
    center_position=(0,0,offset+1.5e-3),
    n=rt.N_BK7.n_function, aperture= r*5
)

lens2_center = (1.5+3/2+756.018+5.4/2)*1e-3

lens2 = rt.ThickRealLens(
    R1 = 0, R2 = -519e-3, center_thickness=5.4e-3,
    center_position=(0,0,offset+lens2_center),
    n=rt.N_BK7.n_function, aperture= r*5
)

screen = rt.Screen(center_position=(0,0,offset+lens2_center+(5.4e-3)/2+50e-3), surfaces=[rt.PlaneSurface(
    center_position=(0,0,lens2_center+(5.4e-3)/2+50e-3), normal=(0,0,-1), aperture_radius = r*5
)])

system = rt.RayOpticalSystem([lens1, lens2, screen])
spectrum = rt.from_wavelength_list(np.array([790,800,810])*1e-9)
#spectrum = rt.gaussian_spectrum_omega(800e-9, 40e-9, num = 21)
laser = rt.RayBundle.collimated_line_spectral(
    np.linspace(-r,r,N_rays),
    0,
    spectrum,

)

fig,ax = plt.subplots()
result = system.trace_and_plot_xz(rays=laser, ax = ax, max_rays=10000, color_style="plasma", wavelength_indizes="all")
plt.show()

#st_summary = rt.spatiotemporal.spatiotemporal_summary(result.rays, phase_order=3)
fig,ax = plt.subplots()
#rt.plot_spectral_phase_against_radius(st= st_summary, ax = ax, phase_parameter="phi0")
print(result.rays.central_beam_index)
print(result.rays.wavelength)
lens1_opl = result.opl_gain_for_element(lens1)
lens2_opl = result.opl_gain_for_element(lens2)
elementopl = result.opl_gain_all_elements()
lens1_phase = result.phase_gain_for_element(lens1)
lens2_phase = result.phase_gain_for_element(lens2)
elementphase_single, elementphase, names = result.phase_gain_all_elements()
ax.plot(result.history[0].radius[0,...]/r,result.rays.phase[0, result.rays.central_beam_index[0,-1]]-result.rays.phase[0,...],"x", label = "790 sim")
#ax.plot(result.rays.radius[0,...]/result.rays.radius[0,...].max(),result.rays.phase[0, result.rays.central_beam_index[0,-1]]-result.rays.phase[0,...],"x", label = "790 sim")
ax.plot(result.history[0].radius[1,...]/r,result.rays.phase[1, result.rays.central_beam_index[result.rays.index_omega0,-1]]-result.rays.phase[1,...],"x", label = "800 sim")
#ax.plot(result.rays.radius[1,...]/result.rays.radius[1,...].max(),result.rays.phase[1, result.rays.central_beam_index[result.rays.index_omega0,-1]]-result.rays.phase[1,...],"x", label = "800 sim")
ax.plot(result.history[0].radius[2,...]/r, result.rays.phase[2, result.rays.central_beam_index[2,-1]]-result.rays.phase[2,...], "x", label = "810 sim")
#ax.plot(result.rays.radius[2,...]/result.rays.radius[2,...].max(), result.rays.phase[2, result.rays.central_beam_index[2,-1]]-result.rays.phase[2,...], "x", label = "810 sim")
ax.plot(puple, wl_790, label = "790 zemax")
ax.plot(puple, wl_800, label = "800 zemax")
ax.plot(puple, wl_810, label = "810 zemax")
# ax.plot(result.rays.radius[0,...]/result.rays.radius[0,...].max(), elementphase[0,...]-elementphase[0, result.rays.central_beam_index[0,-1]],".:", label = "790 sim element only")
# ax.plot(result.rays.radius[1,...]/result.rays.radius[1,...].max(), elementphase[1,...]-elementphase[1, result.rays.central_beam_index[result.rays.index_omega0,-1]],".:", label = "800 sim e.o")
# ax.plot(result.rays.radius[2,...]/result.rays.radius[2,...].max(), elementphase[2,...]-elementphase[2, result.rays.central_beam_index[2,-1]], ".:", label = "810 sim e.o.")

ax.legend()
plt.show()



