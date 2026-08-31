import numpy as np
from dataclasses import dataclass

from ..core.core_classes import (
    RayBundle,
    RayTraceResult,
    element_base,
    FieldBase,
)

from .lenses import ThickRealLens
from ..raytracing.backend.surfaces import (
    check_lens_surface_separation, 
    check_surface_separation_common_frame, 
    SurfaceSeparationCheck
)

@dataclass
class DoubletSeparationCheck:
    valid: bool
    lens1: SurfaceSeparationCheck
    lens2: SurfaceSeparationCheck
    air_gap: SurfaceSeparationCheck

    @property
    def min_lens1_thickness(self) -> float:
        return self.lens1.min_separation

    @property
    def min_lens2_thickness(self) -> float:
        return self.lens2.min_separation

    @property
    def min_air_gap(self) -> float:
        return self.air_gap.min_separation

    def summary(self) -> dict:
        return {
            "valid": self.valid,
            "lens1_valid": self.lens1.valid,
            "lens2_valid": self.lens2.valid,
            "air_gap_valid": self.air_gap.valid,
            "min_lens1_thickness": self.lens1.min_separation,
            "min_lens2_thickness": self.lens2.min_separation,
            "min_air_gap": self.air_gap.min_separation,
            "lens1_r_crit": self.lens1.r_crit,
            "lens2_r_crit": self.lens2.r_crit,
            "air_gap_r_crit": self.air_gap.r_crit,
        }


class GenericAssembly(element_base):
    def __init__(
        self,
        elements: list[element_base],
        distances: list[float] | None = None,
        center_position=None,
        rotation=None,
        parent=None,
        name: str = "generic_assembly",
        keep_existing_global_poses: bool = True,
        radial_symmetric: bool = False,
        n_environment=None,
    ):
        if len(elements) == 0:
            raise ValueError("GenericAssembly requires at least one element.")

        if distances is not None and len(distances) != len(elements) - 1:
            raise ValueError(
                "distances must have length len(elements) - 1."
            )

        super().__init__(
            radial_symmetric=radial_symmetric,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            surfaces=None,
            n_environment=n_environment,
            description="Assembly of multiple optical elements.",
            custom_name=name,
        )

        self.elements = list(elements)
        self.distances = None if distances is None else list(distances)
        self.custom_name = name
        self.name = name

        if keep_existing_global_poses:
            self._capture_children_as_local_poses()
        else:
            if distances is None:
                raise ValueError(
                    "distances must be given when "
                    "keep_existing_global_poses=False."
                )
            self._place_children_from_distances()

    @property
    def _raytracing_available(self):
        return all(
            getattr(element, "_raytracing_available", False)
            for element in self.elements
        )

    @staticmethod
    def _global_center_of(element: element_base) -> np.ndarray:
        return element.local_to_global_points(np.zeros(3, dtype=float))

    @staticmethod
    def _global_rotation_of(element: element_base) -> np.ndarray:
        basis = np.eye(3, dtype=float)
        basis_global = element.local_to_global_directions(basis)
        return basis_global.T

    def _capture_children_as_local_poses(self):
        assembly_global_rotation = self._global_rotation_of(self)

        for element in self.elements:
            child_global_center = self._global_center_of(element)
            child_global_rotation = self._global_rotation_of(element)

            child_local_center = self.global_to_local_points(
                child_global_center
            )

            child_local_rotation = (
                assembly_global_rotation.T @ child_global_rotation
            )

            element.parent = self
            element.set_transform(
                center_position=child_local_center,
                rotation=child_local_rotation,
            )

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        current = rays
        child_results = []

        for element in self.elements:
            result = element.apply(current)
            child_results.append(result)

            if isinstance(result, RayTraceResult):
                current = result.rays
            elif isinstance(result, RayBundle):
                current = result
            else:
                raise TypeError(
                    f"Element {element.name} returned unsupported type "
                    f"{type(result).__name__}."
                )

        try:
            return RayTraceResult(
                rays=current,
                element=self,
                children=child_results,
            )
        except TypeError:
            return RayTraceResult(
                rays=current,
                element=self,
            )

    def _apply_for_wavepropagation(self, field: FieldBase) -> FieldBase:
        current = field

        for element in self.elements:
            current = element.apply(current)

        return current


    def _place_children_from_distances(self):
        """
        Place children along the local assembly z-axis.

        The first element is placed at z=0.
        Each next element is shifted by the corresponding distance.
        Existing child rotations are kept as local rotations.
        """
        if self.distances is None:
            raise ValueError("distances must not be None.")

        z = 0.0

        for i, element in enumerate(self.elements):
            element.parent = self

            element.set_transform(
                center_position=np.array([0.0, 0.0, z], dtype=float),
                rotation=element.rotation,
            )

            if i < len(self.elements) - 1:
                z += float(self.distances[i])

        return self

    def _rebuild_from_distances(self):
        """
        Recompute local child positions from self.distances.

        This preserves each child's local rotation.
        """
        if self.distances is None:
            raise ValueError(
                "This assembly has no distances. "
                "Cannot rebuild distance-based layout."
            )

        if len(self.distances) != len(self.elements) - 1:
            raise ValueError(
                "distances must have length len(elements) - 1."
            )

        z = 0.0

        for i, element in enumerate(self.elements):
            if element.parent is not self:
                element.parent = self

            element.set_transform(
                center_position=np.array([0.0, 0.0, z], dtype=float),
                rotation=element.rotation,
            )

            if i < len(self.elements) - 1:
                z += float(self.distances[i])

        return self

    def set_distance(self, index: int, distance: float):
        """
        Set distance between element[index] and element[index + 1].
        """
        if self.distances is None:
            raise ValueError(
                "This assembly has no distances. "
                "Initialize with distances or call set_distances first."
            )

        if index < 0 or index >= len(self.elements) - 1:
            raise IndexError(
                f"index must be between 0 and {len(self.elements) - 2}."
            )

        self.distances[index] = float(distance)
        return self._rebuild_from_distances()

    def set_distances(self, distances: list[float]):
        """
        Replace all distances and rebuild the assembly.
        """
        if len(distances) != len(self.elements) - 1:
            raise ValueError(
                "distances must have length len(elements) - 1."
            )

        self.distances = [float(d) for d in distances]
        return self._rebuild_from_distances()

    def slide_element(self, element_index: int, dz: float):
        """
        Shift one child element along the assembly-local z-axis.

        This does not modify self.distances.
        """
        if element_index < 0 or element_index >= len(self.elements):
            raise IndexError(
                f"element_index must be between 0 and {len(self.elements) - 1}."
            )

        element = self.elements[element_index]

        pos = element.center_position.copy()
        pos[2] += float(dz)

        element.set_transform(center_position=pos)

        return self


class DoubletAssembly(element_base):
    """
    Rigid assembly of two optical elements interpreted as a lens doublet.

    The doublet itself has a transform through element_base / TransformMixin.
    The two lenses are children of the doublet, so moving or rotating the
    doublet moves both lenses together.

    Geometry convention
    -------------------
    lens1 is placed at local doublet z = 0.

    lens2 is placed such that the air gap between the exit surface of lens1
    and the entrance surface of lens2 equals self.air_gap.

    With the common ThickRealLens convention:

        lens local surface 1: z = 0
        lens local surface 2: z = center_thickness

    the second lens origin is placed at:

        z2 = lens1.center_thickness + air_gap

    Parameters
    ----------
    lens1, lens2:
        Optical elements forming the doublet.

    air_gap:
        Air gap between lens1 exit surface and lens2 entrance surface [m].

    center_position:
        Position of the doublet frame. Global if parent is None, otherwise
        relative to parent.

    rotation:
        Rotation matrix of the doublet frame. Global if parent is None,
        otherwise relative to parent.

    parent:
        Optional parent TransformMixin.

    name:
        Custom name of the doublet.

    keep_lens_rotations:
        If True, keep the current local rotations of the lenses when rebuilding
        geometry. If False, set both local rotations to identity.

    minimum_air_gap:
        Minimum allowed air gap between the two lenses (cannot be 0 or negative). This is enforced when setting air_gap.
        default = 1e-12 m.
    """

    def __init__(
        self,
        lens1: ThickRealLens,
        lens2: ThickRealLens,
        air_gap: float = 0.0,
        center_position=None,
        rotation=None,
        parent=None,
        name: str = "doublet_assembly",
        radial_symmetric: bool = False,
        n_environment=None,
        keep_lens_rotations: bool = True,
        minimum_air_gap: float = 1e-12,
    ):
        if center_position is None:
            center_position = lens1.center_position.copy()
        super().__init__(
            radial_symmetric=radial_symmetric,
            center_position=center_position,
            rotation=rotation,
            parent=parent,
            surfaces=None,
            n_environment=n_environment,
            description="Two-element lens doublet assembly.",
            custom_name=name,
        )

        self.lens1 = lens1
        self.lens2 = lens2
        self.elements = [self.lens1, self.lens2]

        self.air_gap = float(air_gap)
        if self.air_gap < minimum_air_gap:
            self.air_gap = minimum_air_gap
        self.keep_lens_rotations = bool(keep_lens_rotations)

        self.custom_name = name
        self.name = name

        self.rebuild_geometry()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def _raytracing_available(self):
        return all(
            getattr(element, "_raytracing_available", False)
            for element in self.elements
        )

    # ------------------------------------------------------------------
    # Basic geometry helpers
    # ------------------------------------------------------------------

    def _lens_center_thickness(self, lens: ThickRealLens) -> float:
        """
        Return center thickness of a lens-like element.

        Supports common attribute names.
        """
        for attr in (
            "center_thickness",
            "thickness",
            "d",
        ):
            if hasattr(lens, attr):
                return float(getattr(lens, attr))

        raise AttributeError(
            f"{type(lens).__name__} has no center thickness attribute. "
            "Expected one of: center_thickness, thickness, d."
        )

    def _set_lens_center_thickness(
        self,
        lens: ThickRealLens,
        thickness: float,
    ):
        """
        Set center thickness of a lens-like element.

        Also calls a geometry update method if available.
        """
        thickness = float(thickness)

        if hasattr(lens, "set_center_thickness"):
            lens.set_center_thickness(thickness)
        elif hasattr(lens, "center_thickness"):
            lens.center_thickness = thickness
        elif hasattr(lens, "thickness"):
            lens.thickness = thickness
        elif hasattr(lens, "d"):
            lens.d = thickness
        else:
            raise AttributeError(
                f"{type(lens).__name__} has no settable thickness attribute. "
                "Expected set_center_thickness(...), center_thickness, "
                "thickness, or d."
            )

        # Let the lens rebuild its own internal surfaces if it supports it.
        for method_name in (
            "rebuild_geometry",
            "_rebuild_geometry",
            "update_geometry",
            "_update_geometry",
            "_update_surfaces",
        ):
            if hasattr(lens, method_name):
                getattr(lens, method_name)()
                break

        return lens

    def check_separation(
        self,
        n_r: int = 512,
        n_phi: int = 64,
        min_lens_thickness: float = 0.0,
        min_air_gap: float = 0.0,
        aperture_radius: float | None = None,
    ) -> DoubletSeparationCheck:
        """
        Check physical separations inside the doublet.

        Checks
        ------
        1. lens1 internal surface separation: lens1.S1 -> lens1.S2
        2. lens2 internal surface separation: lens2.S1 -> lens2.S2
        3. air gap separation: lens1.S2 -> lens2.S1

        The lens internal checks use check_lens_surface_separation(...).
        The air-gap check uses check_surface_separation_common_frame(...), which is
        parent-child aware and works for rigidly transformed surfaces.

        Parameters
        ----------
        n_r, n_phi:
            Radial and angular sampling resolution.

        min_lens_thickness:
            Minimum allowed internal lens thickness in meters.

        min_air_gap:
            Minimum allowed air gap in meters.

        aperture_radius:
            Aperture radius used for the air-gap check. If None, the smaller
            available aperture of lens1.S2 and lens2.S1 is used.

        Returns
        -------
        DoubletSeparationCheck
        """
        self.rebuild_geometry()

        lens1_check = check_lens_surface_separation(
            self.lens1,
            n_r=n_r,
            n_phi=n_phi,
            min_separation=min_lens_thickness,
        )

        lens2_check = check_lens_surface_separation(
            self.lens2,
            n_r=n_r,
            n_phi=n_phi,
            min_separation=min_lens_thickness,
        )

        if not hasattr(self.lens1, "S2"):
            raise AttributeError("lens1 must have attribute S2.")

        if not hasattr(self.lens2, "S1"):
            raise AttributeError("lens2 must have attribute S1.")

        s_exit = self.lens1.S2
        s_entry = self.lens2.S1

        if aperture_radius is None:
            candidates = []

            if getattr(s_exit, "aperture_radius", None) is not None:
                candidates.append(float(s_exit.aperture_radius))

            if getattr(s_entry, "aperture_radius", None) is not None:
                candidates.append(float(s_entry.aperture_radius))

            if hasattr(self.lens1, "aperture"):
                candidates.append(float(self.lens1.aperture))

            if hasattr(self.lens2, "aperture"):
                candidates.append(float(self.lens2.aperture))

            if len(candidates) == 0:
                raise ValueError(
                    "aperture_radius is None, but no aperture could be inferred "
                    "from lens1.S2, lens2.S1, lens1, or lens2."
                )

            aperture_radius = min(candidates)

        air_gap_check = check_surface_separation_common_frame(
            surface1=s_exit,
            surface2=s_entry,
            aperture_radius=float(aperture_radius),
            n_r=n_r,
            n_phi=n_phi,
            min_separation=min_air_gap,
        )

        valid = (
            lens1_check.valid
            and lens2_check.valid
            and air_gap_check.valid
        )

        return DoubletSeparationCheck(
            valid=valid,
            lens1=lens1_check,
            lens2=lens2_check,
            air_gap=air_gap_check,
        )

    @property
    def lens1_thickness(self) -> float:
        return self._lens_center_thickness(self.lens1)

    @property
    def lens2_thickness(self) -> float:
        return self._lens_center_thickness(self.lens2)

    @property
    def spacing(self) -> float:
        """
        Origin-to-origin spacing between lens1 and lens2.

        With the default convention this is:

            spacing = lens1.center_thickness + air_gap
        """
        return self.lens1_thickness + self.air_gap

    @property
    def total_center_length(self) -> float:
        """
        Approximate total center length of the doublet.

        With the default convention:

            total = lens1_thickness + air_gap + lens2_thickness
        """
        return self.lens1_thickness + self.air_gap + self.lens2_thickness

    # ------------------------------------------------------------------
    # Geometry rebuild
    # ------------------------------------------------------------------

    def rebuild_geometry(self):
        """
        Recompute local child positions from current lens thicknesses and air gap.

        The doublet frame itself is not changed.
        """
        self.lens1.parent = self
        self.lens2.parent = self

        if self.keep_lens_rotations:
            R1 = np.asarray(self.lens1.rotation, dtype=float)
            R2 = np.asarray(self.lens2.rotation, dtype=float)
        else:
            R1 = np.eye(3, dtype=float)
            R2 = np.eye(3, dtype=float)

        z1 = 0.0
        z2 = self.spacing

        self.lens1.set_transform(
            center_position=np.array([0.0, 0.0, z1], dtype=float),
            rotation=R1,
        )

        self.lens2.set_transform(
            center_position=np.array([0.0, 0.0, z2], dtype=float),
            rotation=R2,
        )
        check = self.check_separation(
            n_r=512,
            n_phi=64,
            min_lens_thickness=1e-12,
            min_air_gap=1e-12,
        )
        if not check.valid:
            raise ValueError(
                "Doublet geometry is invalid. "
                f"lens1 valid: {check.lens1.valid}, "
                f"lens2 valid: {check.lens2.valid}, "
                f"air gap valid: {check.air_gap.valid}. "
                f"Minimum lens1 thickness: {check.min_lens1_thickness:.3e} m, "
                f"Minimum lens2 thickness: {check.min_lens2_thickness:.3e} m, "
                f"Minimum air gap: {check.min_air_gap:.3e} m."
            )
        return self

    # ------------------------------------------------------------------
    # Public parameter setters
    # ------------------------------------------------------------------

    def set_air_gap(self, air_gap: float):
        """
        Set the physical air gap between lens1 and lens2, then rebuild geometry.
        """
        self.air_gap = float(air_gap)
        return self.rebuild_geometry()

    def slide_lens2(self, dz: float):
        """
        Change the air gap by dz.

        Positive dz increases the distance between the two lenses.
        """
        self.air_gap += float(dz)
        return self.rebuild_geometry()

    def set_lens1_thickness(self, thickness: float):
        """
        Set lens1 center thickness and move lens2 so the air gap stays constant.
        """
        self._set_lens_center_thickness(self.lens1, thickness)
        return self.rebuild_geometry()

    def set_lens2_thickness(self, thickness: float):
        """
        Set lens2 center thickness.

        This does not move lens2 origin, because the air gap is defined at
        lens2 entrance surface. It still rebuilds geometry for consistency.
        """
        self._set_lens_center_thickness(self.lens2, thickness)
        return self.rebuild_geometry()

    def set_lens_thicknesses(
        self,
        lens1_thickness: float | None = None,
        lens2_thickness: float | None = None,
    ):
        """
        Set one or both lens thicknesses, keeping air gap fixed.
        """
        if lens1_thickness is not None:
            self._set_lens_center_thickness(self.lens1, lens1_thickness)

        if lens2_thickness is not None:
            self._set_lens_center_thickness(self.lens2, lens2_thickness)

        return self.rebuild_geometry()

    # ------------------------------------------------------------------
    # Access aliases
    # ------------------------------------------------------------------

    @property
    def first(self):
        return self.lens1

    @property
    def second(self):
        return self.lens2

    # ------------------------------------------------------------------
    # Apply methods
    # ------------------------------------------------------------------

    def _apply_for_raytracing(self, rays: RayBundle) -> RayTraceResult:
        """
        Apply lens1 and lens2 sequentially to a RayBundle.
        """
        current = rays
        child_results = []

        for element in self.elements:
            result = element.apply(current)
            child_results.append(result)

            if isinstance(result, RayTraceResult):
                current = result.rays
            elif isinstance(result, RayBundle):
                current = result
            else:
                raise TypeError(
                    f"Element {element.name} returned unsupported type "
                    f"{type(result).__name__}."
                )

        try:
            return RayTraceResult(
                rays=current,
                element=self,
                children=child_results,
            )
        except TypeError:
            return RayTraceResult(
                rays=current,
                element=self,
            )

    def _apply_for_wavepropagation(self, field: FieldBase) -> FieldBase:
        """
        Apply lens1 and lens2 sequentially to a wave-propagation field.
        """
        current = field

        for element in self.elements:
            current = element.apply(current)

        return current

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_to_axes_xz(self, ax, **kwargs):
        """
        Plot all child elements into x-z axes.
        """
        for element in self.elements:
            element.plot_to_axes_xz(ax, **kwargs)

        return ax

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> dict:
        """
        Return a compact geometry summary.
        """
        return {
            "name": self.name,
            "air_gap": self.air_gap,
            "lens1_thickness": self.lens1_thickness,
            "lens2_thickness": self.lens2_thickness,
            "spacing": self.spacing,
            "total_center_length": self.total_center_length,
            "center_position": self.center_position.copy(),
            "rotation": self.rotation.copy(),
        }

from typing import NamedTuple
class doublet_config(NamedTuple):
    d1_thickness: float
    d2_thickness: float
    d1_R1: float
    d1_R2: float
    d2_R1: float
    d2_R2: float
    d1_mat: callable
    d2_mat: callable
    gap: float = 0
    name = None

    def create_lenses(self, center_position=(0,0,0), aperture=76.2e-3/2):
        d1 = ThickRealLens(
            R1 = self.d1_R1,
            R2= self.d1_R2,
            center_thickness=self.d1_thickness,
            center_position = center_position,
            n=self.d1_mat.n_function,
            aperture =aperture
        )

        d2 = ThickRealLens(
            R1 = self.d2_R1,
            R2 = self.d2_R2,
            center_thickness=self.d2_thickness,
            center_position=(0,0,d1.center_position[-1]+self.d1_thickness+1e-10),
            n= self.d2_mat.n_function,
            aperture=aperture
        )
        return d1,d2

    def create_doublet(self, center_position=(0,0,0), aperture=76.2e-3/2,
                       **kwargs):
        d1,d2 = self.create_lenses(center_position, aperture)
        name = self.name if self.name is not None else "doublet_assembly"
        return DoubletAssembly(
            lens1 = d1,
            lens2 = d2,
            air_gap = self.gap,
            name = name,
            **kwargs
        )