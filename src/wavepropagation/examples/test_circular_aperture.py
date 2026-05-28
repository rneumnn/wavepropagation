from wavepropagation.elements import CircularAperture
from wavepropagation.sources.monochromaticSource import MonochromaticSource as MS
from wavepropagation.grid import Grid
import matplotlib.pyplot as plt

def test_circular_aperture():
    grid_size = 10e-2
    grid = Grid(2**8, grid_size)
    Laser = MS.gaussian_beam(grid, 800e-9, w0=1e-3)

    rel_width = .9
    s = .2
    #test circular aperture
    aperture_test = CircularAperture(radius=grid_size*rel_width/2, smoothness=s*grid_size, edge_level=1e-5)
    transmission = aperture_test.transmission(Laser)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    a=ax[0].imshow(transmission, cmap="gray", origin="lower", extent=(-grid_size/2,grid_size/2,-grid_size/2,grid_size/2), norm="log")
    ax[0].set_title("circular aperture transmission")
    plt.colorbar(a, ax=ax[0], label="Transmission (log scale)")
    # dibujar una linea vertical en radius=grid_size[0]*0.9/2
    plt.axvline(x=rel_width*grid_size/2, color='red', linestyle='--')
    plt.axvline(x=rel_width*grid_size/2-s*grid_size, color='blue', linestyle='--')
    plt.axvline(x=-(rel_width*grid_size/2), color='red', linestyle='--')
    plt.axvline(x=-(rel_width*grid_size/2-s*grid_size), color='blue', linestyle='--')
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    # plot the intensity in the axis of the beam
    ax[1].plot(grid.x, transmission[transmission.shape[0]//2,:])
    ax[1].set_title("circular aperture transmission")
    ax[1].set_xlabel("x [m]")
    ax[1].set_ylabel("intensity")
    plt.show()

if __name__ == "__main__":
    test_circular_aperture()    