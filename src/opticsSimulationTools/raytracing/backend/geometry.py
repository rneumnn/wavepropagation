import numpy as np
from dataclasses import dataclass, field



def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize vector or array of vectors.

    Works for shape (3,) and (..., 3).
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(norm < eps, 1.0, norm)

@dataclass
class Line:
    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.direction = normalize(np.asarray(self.direction, dtype=float))

    def is_normalized(self, atol: float = 1e-12) -> bool:
        return np.isclose(np.linalg.norm(self.direction), 1.0, atol=atol)

    def solve_parameter(self, p: np.ndarray) -> float:
        """
        Return the line parameter t for the closest point on the line.

        For normalized direction:
            closest_point = position + t * direction
            t = dot(p - position, direction)
        """
        p = np.asarray(p, dtype=float)

        return float(np.dot(p - self.position, self.direction))

    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate line point at parameter t.
        """
        return self.position + t * self.direction

    def closest_point(self, p: np.ndarray) -> np.ndarray:
        """
        Return closest point on the infinite line to point p.
        """
        t = self.solve_parameter(p)
        return self.evaluate(t)

    def distance_to_point(self, p: np.ndarray) -> float:
        """
        Perpendicular distance from point p to the infinite line.
        """
        p = np.asarray(p, dtype=float)
        closest = self.closest_point(p)
        return float(np.linalg.norm(p - closest))

    def contains(self, p: np.ndarray, atol: float = 1e-12) -> bool:
        """
        Check whether point p lies on the infinite line.
        """
        return self.distance_to_point(p) <= atol
    

@dataclass
class Plane:
    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float)
    )
    normal: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.normal = normalize(np.asarray(self.normal, dtype=float))

    @property
    def coordinate_form(self) -> np.ndarray:
        """
        Plane coordinate form:

            a*x + b*y + c*z + d = 0

        Returns
        -------
        coeffs:
            np.ndarray [a, b, c, d]
        """
        d = -np.dot(self.normal, self.position)
        return np.array([*self.normal, d], dtype=float)

    def signed_distance(self, p: np.ndarray) -> float:
        """
        Signed distance from point to plane.

        Since normal is normalized, this has units of meters.
        """
        p = np.asarray(p, dtype=float)
        return float(np.dot(self.normal, p - self.position))

    def distance_to_point(self, p: np.ndarray) -> float:
        """
        Absolute distance from point to plane.
        """
        return abs(self.signed_distance(p))

    def contains(self, p: np.ndarray, atol: float = 1e-12) -> bool:
        """
        Check whether point p lies on the plane.
        """
        return self.distance_to_point(p) <= atol

    def project_point(self, p: np.ndarray) -> np.ndarray:
        """
        Orthogonal projection of point p onto the plane.
        """
        p = np.asarray(p, dtype=float)
        return p - self.signed_distance(p) * self.normal

    @property
    def parameter_form(self):
        """
        Return a parameterized plane:

            r = a + s*b + t*c

        where:
            a = point on plane
            b, c = two orthonormal direction vectors in the plane

        Returns
        -------
        np.ndarray shape (3, 3):
            [a, b, c]
        """
        n = self.normal

        # Pick a vector that is not parallel to n.
        if abs(n[0]) < 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        else:
            helper = np.array([0.0, 1.0, 0.0])

        b = normalize(np.cross(n, helper))
        c = normalize(np.cross(n, b))

        a = self.position

        return np.array([a, b, c], dtype=float)
    
    def intersect_line_parameter(self, line: Line, atol: float = 1e-12):
        denom = np.dot(self.normal, line.direction)

        if abs(denom) < atol:
            return np.nan, False

        t = np.dot(self.normal, self.position - line.position) / denom
        return t, True
       
def intersect_planes(p1: Plane, p2: Plane, atol: float = 1e-12) -> Line:
    """
    Return the intersection line of two planes.

    Parameters
    ----------
    p1, p2:
        Plane objects.

    Returns
    -------
    line:
        Line object representing the intersection.

    Raises
    ------
    ValueError:
        If planes are parallel or coincident.
    """
    c1 = p1.coordinate_form
    c2 = p2.coordinate_form

    n1 = c1[:3]
    n2 = c2[:3]

    d1 = c1[3]
    d2 = c2[3]

    # Direction of intersection line.
    direction = np.cross(n1, n2)
    norm_direction = np.linalg.norm(direction)

    if norm_direction < atol:
        # Planes are parallel. Could be identical or separate.
        if abs(p1.signed_distance(p2.position)) <= atol:
            raise ValueError("Planes are coincident; intersection is the whole plane.")
        raise ValueError("Planes are parallel; no intersection line.")

    direction = direction / norm_direction

    # Find one point on the intersection line.
    #
    # We set the coordinate with the largest direction component to zero.
    # Then solve the remaining 2x2 system.
    fixed_axis = int(np.argmax(np.abs(direction)))

    A = np.delete(np.vstack([n1, n2]), fixed_axis, axis=1)
    b = -np.array([d1, d2], dtype=float)

    partial = np.linalg.solve(A, b)

    point = np.insert(partial, fixed_axis, 0.0)

    return Line(position=point, direction=direction)

def intersect_line_plane(line: Line, plane: Plane, atol: float = 1e-12):
    """
    Intersect infinite line with plane.

    Returns
    -------
    point:
        Intersection point.

    t:
        Line parameter.

    Raises
    ------
    ValueError:
        If line is parallel to plane.
    """
    denom = np.dot(plane.normal, line.direction)

    if abs(denom) < atol:
        if plane.contains(line.position, atol=atol):
            raise ValueError("Line lies in the plane.")
        raise ValueError("Line is parallel to the plane.")

    t = np.dot(plane.normal, plane.position - line.position) / denom
    point = line.evaluate(t)

    return point, t
        

def orient_normal_against_ray(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Flip normals so that they point against the incoming ray direction.

    direction: (..., 3) - RayBundle direction vectors
    normal:    (..., 3) - Normal vectors of surface at the intersection points
    """
    dot = np.sum(direction * normal, axis=-1)

    # If dot > 0, normal points roughly in same direction as ray.
    # Flip it so it points against the ray.
    flip = dot > 0

    return np.where(flip[..., None], -normal, normal)


def vector_from_angles(phi, theta) -> np.ndarray:
    """
    Create a unit vector from spherical-like angles.

    phi:
        Deflection angle from the y-z plane toward +x.
        phi = 0 means the vector lies in the y-z plane.
        phi = +pi/2 means the vector points along +x.
        phi = -pi/2 means the vector points along -x.

    theta:
        Angle inside the y-z plane, measured from +z toward +y.
        theta = 0 means +z.
        theta = +pi/2 means +y.
        theta = pi means -z.
        theta = -pi/2 means -y.

    Returns
    -------
    np.ndarray:
        Unit vector [x, y, z].
    """
    x = np.sin(phi)
    y = np.cos(phi) * np.sin(theta)
    z = np.cos(phi) * np.cos(theta)

    return np.array([x, y, z], dtype=float)