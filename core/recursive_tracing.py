import numpy as np

from shadow_ray import generate_shadow_ray
generate_shadow_ray = generate_shadow_ray()

from ray_intersection import generate_reflected_ray
generate_reflected_ray = generate_reflected_ray()

from ray_intersection import generate_refracted_ray
generate_refracted_ray = generate_refracted_ray()

from render_equation import render_equation
render_equation = render_equation()

def recursive_tracing(ray, scene, depth):
    """Recursively traces a ray through a scene.

    Args:
        ray (np.array): The ray to trace.
        scene (Scene): The scene to trace the ray through.
        depth (int): The current recursion depth.

    Returns:
        np.array: A 3D array of the color of the pixel.
    """
    # Check if the maximum recursion depth has been reached
    if depth > scene.max_depth:
        return np.array([0, 0, 0])

    # Find the closest intersection point
    closest_intersection = scene.find_closest_intersection(ray)

    # Check if an intersection was found
    if closest_intersection is None:
        return np.array([0, 0, 0])

    # Calculate the color of the pixel
    color = np.array([0, 0, 0])
    for light in scene.lights:
        # Generate the light ray
        light_ray = light.generate_light_ray(closest_intersection[0])

        # Generate the shadow ray
        shadow_ray = generate_shadow_ray(light.position, closest_intersection[0])

        # Calculate the color of the pixel using the render equation
        color += render_equation(light_ray, shadow_ray, closest_intersection[0], closest_intersection[1])

    # Calculate the reflected color
    reflected_ray = generate_reflected_ray(ray, closest_intersection[0], closest_intersection[2])
    reflected_color = recursive_tracing(reflected_ray, scene, depth + 1)

    # Calculate the refracted color
    refracted_ray = generate_refracted_ray(ray, closest_intersection[0], closest_intersection[2], closest_intersection[1].refractive_index)
    refracted_color = recursive_tracing(refracted_ray, scene, depth + 1)

    # Calculate the final color
    color += reflected_color * closest_intersection[1].reflection_coefficient
    color += refracted_color * closest_intersection[1].refraction_coefficient

    return color