import numpy as np

# Camera Ray

def generate_camera_ray(camera_position, camera_direction, camera_fov, image_width, image_height):
    """Generates a camera ray for each pixel in the image.

    Args:
        camera_position (np.array): The position of the camera in 3D space.
        camera_direction (np.array): The direction the camera is pointing in 3D space.
        camera_fov (float): The field of view of the camera.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.

    Returns:
        np.array: A 3D array of camera rays for each pixel in the image.
    """
    # Calculate the camera's view plane
    view_plane_width = 2 * np.tan(camera_fov / 2)
    view_plane_height = view_plane_width * (image_height / image_width)

    # Generate the camera rays
    camera_rays = np.zeros((image_height, image_width, 3))
    for y in range(image_height):
        for x in range(image_width):
            # Calculate the ray direction
            ray_direction = camera_direction + (view_plane_width * (x / image_width) - (view_plane_width / 2)) * np.cross(camera_direction, np.array([0, 1, 0])) + (view_plane_height * (y / image_height) - (view_plane_height / 2)) * np.array([0, 1, 0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)

            # Store the ray
            camera_rays[y, x] = np.array([camera_position, ray_direction])

    return camera_rays


# Shadow Ray

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


# Render Equation

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


# Recursive Tracing

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


# Ray Intersection

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