import numpy as np
import pytest
import matplotlib.pyplot as plt

from opticsSimulationTools.raytracing.frontend import (
    RayBundle,
    RayOpticalSystem,
    ThickRealLens,
    PlaneMirror,
    SphericalMirror,
    Axiparabola,
    Screen,
    Prism,
    BK7,
)

from opticsSimulationTools.raytracing.backend.geometry import rotation_matrix_y


def assert_close(a, b, atol=1e-12):
    assert np.allclose(np.asarray(a), np.asarray(b), atol=atol), f"{a} != {b}"


def make_test_rays(z=0.0, wavelength=800e-9, x_max=5e-3, n=21):
    return RayBundle.collimated_line(
        x=np.linspace(-x_max, x_max, n),
        z=z,
        wavelength=wavelength,
    )


def test_thick_real_lens_child_surfaces_follow_translation():
    lens = ThickRealLens(
        R1=0.1,
        R2=-0.1,
        center_thickness=5e-3,
        n=BK7.n_function,
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture=10e-3,
    )

    assert_close(lens.S1.center_position, [0.0, 0.0, 0.0])
    assert_close(lens.S2.center_position, [0.0, 0.0, 5e-3])

    assert_close(lens.S1.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.2])
    assert_close(lens.S2.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.205])

    lens.set_transform(center_position=np.array([0.0, 0.0, 0.5]))

    assert_close(lens.S1.center_position, [0.0, 0.0, 0.0])
    assert_close(lens.S2.center_position, [0.0, 0.0, 5e-3])

    assert_close(lens.S1.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.5])
    assert_close(lens.S2.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.505])


def test_thick_real_lens_raytrace_unrotated_and_rotated():
    rays = make_test_rays()

    lens = ThickRealLens(
        R1=0.1,
        R2=-0.1,
        center_thickness=5e-3,
        n=BK7.n_function,
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture=10e-3,
    )

    result = lens.apply(rays)

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)

    lens.set_transform(
        center_position=np.array([0.0, 0.0, 0.2]),
        rotation=rotation_matrix_y(np.deg2rad(5.0)),
    )

    result_rot = lens.apply(rays)

    assert result_rot.rays.positions.shape == result_rot.rays.directions.shape
    assert result_rot.rays.valid.shape == result_rot.rays.shape
    assert np.any(result_rot.rays.valid)


def test_plane_mirror_child_surface_follows_translation_and_rotation():
    mirror = PlaneMirror.from_euler_deg(
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture_radius=10e-3,
        ry_deg=45.0,
    )

    assert_close(mirror.surface.center_position, [0.0, 0.0, 0.0])
    assert_close(mirror.surface.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.2])

    rays = make_test_rays()
    result = mirror.apply(rays)

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)

    # Reflected directions should not be identical to input directions.
    assert not np.allclose(result.rays.directions, rays.directions)


def test_spherical_mirror_raytrace_parent_child():
    mirror = SphericalMirror.from_euler_deg(
        center_position=np.array([0.0, 0.0, 0.2]),
        R=0.2,
        aperture_radius=10e-3,
        ry_deg=5.0,
    )

    rays = make_test_rays()
    result = mirror.apply(rays)

    assert_close(mirror.surface.center_position, [0.0, 0.0, 0.0])
    assert_close(mirror.surface.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.2])

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)


def test_axiparabola_raytrace_parent_child():
    mirror = Axiparabola.from_euler_deg(
        F0=0.2,
        L=0.02,
        aperture_radius=10e-3,
        center_position=np.array([0.0, 0.0, 0.2]),
        ry_deg=5.0,
    )

    rays = make_test_rays(x_max=4e-3)
    result = mirror.apply(rays)

    assert_close(mirror.surface.center_position, [0.0, 0.0, 0.0])
    assert_close(mirror.surface.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.2])

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)


def test_screen_parent_child():
    screen = Screen.from_euler_deg(
        center_position=np.array([0.0, 0.0, 0.5]),
        ry_deg=10.0,
    )

    assert_close(screen.surface.center_position, [0.0, 0.0, 0.0])
    assert_close(screen.surface.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.5])

    rays = make_test_rays()
    result = screen.apply(rays)

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)


def test_prism_child_surfaces_follow_translation():
    prism = Prism(
        surface1_angles=(100.0, 0.0),
        surface2_angles=(-70.0, 0.0),
        center_thickness=10e-3,
        material=BK7.n_function,
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture_radius=10e-3,
    )

    assert_close(prism.S1.center_position, [0.0, 0.0, -5e-3])
    assert_close(prism.S2.center_position, [0.0, 0.0, 5e-3])

    assert_close(prism.S1.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.195])
    assert_close(prism.S2.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 0.205])

    prism.set_transform(center_position=np.array([0.0, 0.0, 50e-3]))

    assert_close(prism.S1.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 45e-3])
    assert_close(prism.S2.local_to_global_points([0.0, 0.0, 0.0]), [0.0, 0.0, 55e-3])


def test_prism_thickness_positive():
    prism = Prism(
        surface1_angles=(100.0, 0.0),
        surface2_angles=(-70.0, 0.0),
        center_thickness=10e-3,
        material=BK7.n_function,
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture_radius=10e-3,
    )

    x = np.linspace(-5e-3, 5e-3, 21)
    thickness = prism.thickness_at_x(x)

    assert np.all(np.isfinite(thickness))
    assert np.any(thickness > 0.0)


def test_prism_raytrace_unrotated_and_rotated():
    rays = make_test_rays(x_max=5e-3)

    prism = Prism(
        surface1_angles=(100.0, 0.0),
        surface2_angles=(-70.0, 0.0),
        center_thickness=10e-3,
        material=BK7.n_function,
        center_position=np.array([0.0, 0.0, 0.2]),
        aperture_radius=10e-3,
    )

    result = prism.apply(rays)

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)

    prism.set_transform(
        center_position=np.array([0.0, 0.0, 0.2]),
        rotation=rotation_matrix_y(np.deg2rad(5.0)),
    )

    result_rot = prism.apply(rays)

    assert result_rot.rays.positions.shape == result_rot.rays.directions.shape
    assert result_rot.rays.valid.shape == result_rot.rays.shape
    assert np.any(result_rot.rays.valid)


def test_prism_plot_does_not_raise():
    prism = Prism(
        surface1_angles=(100.0, 0.0),
        surface2_angles=(-70.0, 0.0),
        center_thickness=10e-3,
        material=BK7.n_function,
        center_position=np.array([0.0, 0.0, 50e-3]),
        aperture_radius=10e-3,
    )

    fig, ax = plt.subplots()

    try:
        prism.plot_to_axes_xz(ax)
    finally:
        plt.close(fig)


def test_ray_optical_system_prism_to_screen_trace():
    rays = make_test_rays(x_max=5e-3)

    prism = Prism(
        surface1_angles=(100.0, 0.0),
        surface2_angles=(-70.0, 0.0),
        center_thickness=10e-3,
        material=BK7.n_function,
        center_position=np.array([0.0, 0.0, 50e-3]),
        aperture_radius=10e-3,
    )

    screen = Screen.FlatScreen(
        center_position=np.array([0.0, 0.0, 100e-3]),
    )

    system = RayOpticalSystem((prism, screen))
    result = system.trace(rays)

    assert result.rays.positions.shape == result.rays.directions.shape
    assert result.rays.valid.shape == result.rays.shape
    assert np.any(result.rays.valid)