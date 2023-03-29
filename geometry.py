# Sphere Intersection
import numpy as np

def sphere_intersection(ray, sphere_position, sphere_radius):
    """Calculates the intersection of a ray with a sphere.

    Args:
        ray (np.array): The ray to intersect with.
        sphere_position (np.array): The position of the sphere in 3D space.
        sphere_radius (float): The radius of the sphere.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the distance from the ray origin to the center of the sphere
    distance = np.linalg.norm(sphere_position - ray[0])

    # Calculate the intersection point
    if distance <= sphere_radius:
        point = ray[0] + ray[1] * (sphere_radius - distance)
        return (point, 'sphere', sphere_position - point)
    else:
        return None

# Plane Intersection

def plane_intersection(ray, plane_normal, plane_position):
    """Calculates the intersection of a ray with a plane.

    Args:
        ray (np.array): The ray to intersect with.
        plane_normal (np.array): The normal of the plane in 3D space.
        plane_position (np.array): The position of the plane in 3D space.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the intersection point
    t = (plane_position - ray[0]) @ plane_normal / (ray[1] @ plane_normal)
    if t >= 0:
        point = ray[0] + ray[1] * t
        return (point, 'plane', plane_normal)
    else:
        return None

# Triangle Intersection

def triangle_intersection(ray, triangle_vertices):
    """Calculates the intersection of a ray with a triangle.

    Args:
        ray (np.array): The ray to intersect with.
        triangle_vertices (np.array): The vertices of the triangle in 3D space.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the intersection point
    v0 = triangle_vertices[1] - triangle_vertices[0]
    v1 = triangle_vertices[2] - triangle_vertices[0]
    v2 = ray[0] - triangle_vertices[0]

    # Calculate the barycentric coordinates
    d = np.linalg.det([v0, v1, ray[1]])
    barycentric_coordinates = np.linalg.det([v2, v1, ray[1]]) / d
    if barycentric_coordinates < 0:
        return None
    else:
        barycentric_coordinates = np.linalg.det([v0, v2, ray[1]]) / d
        if barycentric_coordinates < 0:
            return None
        else:
            t = np.linalg.det([v0, v1, v2]) / d
            if t >= 0:
                point = ray[0] + ray[1] * t
                return (point, 'triangle', np.cross(v0, v1))
            else:
                return None

# Quad Intersection

def quad_intersection(ray, quad_vertices):
    """Calculates the intersection of a ray with a quad.

    Args:
        ray (np.array): The ray to intersect with.
        quad_vertices (np.array): The vertices of the quad in 3D space.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the intersection point
    u = quad_vertices[1] - quad_vertices[0]
    v = quad_vertices[3] - quad_vertices[0]
    w = ray[0] - quad_vertices[0]

    # Calculate the barycentric coordinates
    d = np.linalg.det([u, v, ray[1]])
    barycentric_coordinates = np.linalg.det([w, v, ray[1]]) / d
    if barycentric_coordinates < 0 or barycentric_coordinates > 1:
        return None
    else:
        barycentric_coordinates = np.linalg.det([u, w, ray[1]]) / d
        if barycentric_coordinates < 0 or barycentric_coordinates > 1:
            return None
        else:
            t = np.linalg.det([u, v, w]) / d
            if t >= 0:
                point = ray[0] + ray[1] * t
                return (point, 'quad', np.cross(u, v))
            else:
                return None

# Cube Intersection

def cube_intersection(ray, cube_vertices):
    """Calculates the intersection of a ray with a cube.

    Args:
        ray (np.array): The ray to intersect with.
        cube_vertices (np.array): The vertices of the cube in 3D space.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the intersection point
    u = cube_vertices[1] - cube_vertices[0]
    v = cube_vertices[3] - cube_vertices[0]
    w = ray[0] - cube_vertices[0]

    # Calculate the barycentric coordinates
    d = np.linalg.det([u, v, ray[1]])
    barycentric_coordinates = np.linalg.det([w, v, ray[1]]) / d
    if barycentric_coordinates < 0 or barycentric_coordinates > 1:
        return None
    else:
        barycentric_coordinates = np.linalg.det([u, w, ray[1]]) / d
        if barycentric_coordinates < 0 or barycentric_coordinates > 1:
            return None
        else:
            t = np.linalg.det([u, v, w]) / d
            if t >= 0:
                point = ray[0] + ray[1] * t
                return (point, 'cube', np.cross(u, v))
            else:
                return None

# Cone Intersection

def cone_intersection(ray, cone_position, cone_direction, cone_angle):
    """Calculates the intersection of a ray with a cone.

    Args:
        ray (np.array): The ray to intersect with.
        cone_position (np.array): The position of the cone in 3D space.
        cone_direction (np.array): The direction of the cone in 3D space.
        cone_angle (float): The angle of the cone.

    Returns:
        tuple: A tuple containing the intersection point, the material of the intersected object, and the normal of the surface at the intersection point. If no intersection is found, returns None.
    """
    # Calculate the intersection point
    a = np.dot(ray[1], cone_direction)**2 - np.cos(cone_angle)**2
    b = 2 * np.dot(ray[1], cone_direction) * (np.dot(ray[0], cone_direction) - np.dot(cone_position, cone_direction))
    c = np.dot(ray[0], cone_direction)**2 - np.dot(cone_position, cone_direction)**2
    discriminant = b**2 - 4*a*c

    # Calculate the points of intersection
    if discriminant >= 0:
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        if t1 > 0 and t2 > 0:
            # Take the nearest point of intersection
            if t1 < t2:
                point = ray[0] + ray[1] * t1
            else:
                point = ray[0] + ray[1] * t2
            return (point, 'cone', cone_direction)
        else:
            return None
    else:
        return None