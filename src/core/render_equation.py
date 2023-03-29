import numpy as np

def render_equation(light_ray, shadow_ray, intersection_point, material):
    """Calculates the color of a pixel using the render equation.

    Args:
        light_ray (np.array): The ray from the light source to the intersection point.
        shadow_ray (np.array): The ray from the intersection point to the light source.
        intersection_point (np.array): The intersection point in 3D space.
        material (Material): The material of the object at the intersection point.

    Returns:
        np.array: A 3D array of the color of the pixel.
    """
    # Calculate the color of the pixel
    color = material.diffuse_color * material.diffuse_coefficient * np.dot(light_ray[1], shadow_ray[1])

    return color 