import numpy as np

def generate_shadow_ray(light_position, intersection_point):
    """Generates a shadow ray from an intersection point to a light source.

    Args:
        light_position (np.array): The position of the light source in 3D space.
        intersection_point (np.array): The intersection point in 3D space.

    Returns:
        np.array: A 3D array of the shadow ray.
    """
    # Calculate the ray direction
    ray_direction = light_position - intersection_point
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    # Store the ray
    shadow_ray = np.array([intersection_point, ray_direction])

    return shadow_ray