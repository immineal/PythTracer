# Diffuse Lambert Material

import numpy as np
import cupy
from light import calculate_radiance_unit
from core import calculate_normal, generate_reflected_ray

def diffuse_lambert(light_ray, shadow_ray, intersection_point, material):
    """Calculates the diffuse reflection for a pixel using the Lambertian model.

    Args:
        light_ray (np.array): The ray from the intersection point to the light source.
        shadow_ray (np.array): The ray from the intersection point to the shadow source.
        intersection_point (np.array): The 3D intersection point of the ray and the object.
        material (dict): The material properties for the object.

    Returns:
        np.array: The color of the pixel.
    """

    # Get the material color
    material_color = material['color']

    # Get the light color
    light_color = material['light_color']

    # Calculate the radiance unit
    radiance_unit = calculate_radiance_unit(light_ray, light_color, material_color)

    # Calculate the diffuse reflection with the Lambertian model
    diffuse_reflection = (light_color * material_color * radiance_unit) / np.linalg.norm(light_ray)

    # Check if the intersection point is in shadow
    if np.linalg.norm(shadow_ray) > 0:
        # If in shadow, set the diffuse reflection to 0
        diffuse_reflection = np.zeros(3)

    # Return the diffuse reflection
    return diffuse_reflection

# Mirror Material

def mirror(light_ray, shadow_ray, intersection_point, material):
    """Calculates the mirror reflection for a pixel.

    Args:
        light_ray (np.array): The ray from the intersection point to the light source.
        shadow_ray (np.array): The ray from the intersection point to the shadow source.
        intersection_point (np.array): The 3D intersection point of the ray and the object.
        material (dict): The material properties for the object.

    Returns:
        np.array: The color of the pixel.
    """

    # Get the material color
    material_color = material['color']

    # Get the light color
    light_color = material['light_color']

    # Calculate the normal at the intersection point
    normal = calculate_normal(intersection_point, material)

    # Calculate the reflected ray
    reflected_ray = generate_reflected_ray(light_ray, intersection_point, normal)

    # Calculate the radiance unit
    radiance_unit = calculate_radiance_unit(light_ray, light_color, material_color)

    # Calculate the mirror reflection
    mirror_reflection = (light_color * material_color * radiance_unit) * np.linalg.norm(reflected_ray)

    # Check if the intersection point is in shadow
    if np.linalg.norm(shadow_ray) > 0:
        # If in shadow, set the mirror reflection to 0
        mirror_reflection = np.zeros(3)

    # Return the mirror reflection
    return mirror_reflection

# Glossy Material

def glossy(light_ray, shadow_ray, intersection_point, material):
    """Calculates the glossy reflection for a pixel.

    Args:
        light_ray (np.array): The ray from the intersection point to the light source.
        shadow_ray (np.array): The ray from the intersection point to the shadow source.
        intersection_point (np.array): The 3D intersection point of the ray and the object.
        material (dict): The material properties for the object.

    Returns:
        np.array: The color of the pixel.
    """

    # Get the material color
    material_color = material['color']

    # Get the light color
    light_color = material['light_color']

    # Get the roughness
    roughness = material['roughness']

    # Calculate the normal at the intersection point
    normal = calculate_normal(intersection_point, material)

    # Create a random vector for the glossy reflection
    random_vector = cupy.random.rand(3)

    # Calculate the glossy reflection
    glossy_reflection = (light_color * material_color * roughness * random_vector * np.linalg.norm(light_ray)) / np.linalg.norm(normal)

    # Check if the intersection point is in shadow
    if np.linalg.norm(shadow_ray) > 0:
        # If in shadow, set the glossy reflection to 0
        glossy_reflection = np.zeros(3)

    # Return the glossy reflection
    return glossy_reflection

# Metal Material

def metal(light_ray, shadow_ray, intersection_point, material):
    """Calculates the metal reflection for a pixel.

    Args:
        light_ray (np.array): The ray from the intersection point to the light source.
        shadow_ray (np.array): The ray from the intersection point to the shadow source.
        intersection_point (np.array): The 3D intersection point of the ray and the object.
        material (dict): The material properties for the object.

    Returns:
        np.array: The color of the pixel.
    """

    # Get the material color
    material_color = material['color']

    # Get the light color
    light_color = material['light_color']

    # Get the metalness
    metalness = material['metalness']

    # Calculate the normal at the intersection point
    normal = calculate_normal(intersection_point, material)

    # Calculate the reflected ray
    reflected_ray = generate_reflected_ray(light_ray, intersection_point, normal)

    # Calculate the metal reflection
    metal_reflection = (light_color * material_color * metalness * np.linalg.norm(reflected_ray)) / np.linalg.norm(light_ray)

    # Check if the intersection point is in shadow
    if np.linalg.norm(shadow_ray) > 0:
        # If in shadow, set the metal reflection to 0
        metal_reflection = np.zeros(3)

    # Return the metal reflection
    return metal_reflection