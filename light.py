# Point Light

import numpy as np
from geometry import sphere_intersection, plane_intersection

def generate_point_light_ray(light_position, intersection_point):
    """Generates a ray from an intersection point to a point light source.

    Args:
        light_position (np.array): The position of the point light.
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the point light.
    """
    light_ray = light_position - intersection_point
    return light_ray

def point_light_intersection(light_position, ray):
    """Calculates the intersection of a ray with a point light.

    Args:
        light_position (np.array): The position of the point light.
        ray (np.array): The ray to intersect with.

    Returns:
        np.array: A 3D array of the intersection point with the point light.
    """
    intersection = sphere_intersection(ray, light_position, 0)
    return intersection

# Area Light

def generate_area_light_ray(light_position, intersection_point):
    """Generates a ray from an intersection point to an area light source.

    Args:
        light_position (np.array): The position of the area light.
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the area light.
    """
    light_ray = light_position - intersection_point
    return light_ray

def area_light_intersection(light_position, ray):
    """Calculates the intersection of a ray with an area light.

    Args:
        light_position (np.array): The position of the area light.
        ray (np.array): The ray to intersect with.

    Returns:
        np.array: A 3D array of the intersection point with the area light.
    """
    intersection = plane_intersection(ray, light_position, light_position)
    return intersection

# Radiance Unit

def calculate_radiance_unit(light_ray, light_color, material_color):
    """Calculates the radiance unit for a pixel.

    Args:
        light_ray (np.array): The ray from the intersection point to the light source.
        light_color (np.array): The color of the light source.
        material_color (np.array): The color of the material.

    Returns:
        np.array: A 3D array of the radiance unit for the pixel.
    """
    light_ray_norm = np.linalg.norm(light_ray)
    radiance_unit = light_color * material_color / (4 * np.pi * light_ray_norm**2)
    return radiance_unit

# Directional Light

def generate_directional_light_ray(light_direction, intersection_point):
    """Generates a ray from an intersection point to a directional light source.

    Args:
        light_direction (np.array): The direction of the directional light.
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the directional light.
    """
    light_ray = light_direction
    return light_ray

# Spot Light

def generate_spot_light_ray(light_position, light_direction, intersection_point):
    """Generates a ray from an intersection point to a spot light source.

    Args:
        light_position (np.array): The position of the spot light.
        light_direction (np.array): The direction of the spot light.
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the spot light.
    """
    light_ray = light_position - intersection_point
    return light_ray

# Environment Light

def generate_environment_light_ray(intersection_point):
    """Generates a ray from an intersection point to an environment light source.

    Args:
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the environment light.
    """
    light_ray = np.array([1, 0, 0])
    return light_ray

# Image Light

def generate_image_based_light_ray(light_image, intersection_point):
    """Generates a ray from an intersection point to an image based light source.

    Args:
        light_image (np.array): The image representing the light source.
        intersection_point (np.array): The point of intersection with a surface.

    Returns:
        np.array: A 3D array of rays from the intersection point to the image based light.
    """
    # Get the pixel coordinates from the intersection point
    x_coord, y_coord = intersection_point[0], intersection_point[1]
    # Get the corresponding pixel from the image
    pixel = light_image[x_coord, y_coord]
    # Construct the light ray
    light_ray = np.array([pixel[2], pixel[1], pixel[0]])
    return light_ray