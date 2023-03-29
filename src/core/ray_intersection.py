import numpy as np

def generate_reflected_ray(ray, intersection_point, normal):
    """Generates a reflected ray from an intersection point.

    Args:
        ray (np.array): The ray to reflect.
        intersection_point (np.array): The intersection point in 3D space.
        normal (np.array): The normal of the surface at the intersection point.

    Returns:
        np.array: A 3D array of the reflected ray.
    """
    # Calculate the ray direction
    ray_direction = ray[1] - 2 * np.dot(ray[1], normal) * normal

    # Store the ray
    reflected_ray = np.array([intersection_point, ray_direction])

    return reflected_ray


def generate_refracted_ray(ray, intersection_point, normal, refractive_index):
    """Generates a refracted ray from an intersection point.

    Args:
        ray (np.array): The ray to refract.
        intersection_point (np.array): The intersection point in 3D space.
        normal (np.array): The normal of the surface at the intersection point.
        refractive_index (float): The refractive index of the material.

    Returns:
        np.array: A 3D array of the refracted ray.
    """
    # Calculate the ray direction
    ray_direction = ray[1] - (1 - refractive_index**2) * np.dot(ray[1], normal) * normal

    # Store the ray
    refracted_ray = np.array([intersection_point, ray_direction])

    return refracted_ray