# Render Window
import numpy as np
from PyQt5.QtWidgets import QMainWindow
from core import generate_camera_ray
from core import generate_shadow_ray
from core import render_equation
from core import recursive_tracing


def render_image(self, camera_position, camera_direction, camera_fov, image_width, image_height, scene, depth):
    """Renders an image using the Pythtracer.

    Args:
        camera_position (np.array): The position of the camera in 3D space.
        camera_direction (np.array): The direction the camera is pointing in 3D space.
        camera_fov (float): The field of view of the camera.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        scene (list): A list of all the objects in the scene.
        depth (int): The maximum recursion depth for the raytracer.

    Returns:
        np.array: A 3D array of RGB values for each pixel in the image.
    """

    # Generate camera rays
    camera_rays = generate_camera_ray(camera_position, camera_direction, camera_fov, image_width, image_height)

    # Generate shadow rays
    shadow_rays = generate_shadow_ray(camera_rays, scene)

    # Generate image buffer
    image_buffer = np.zeros((image_height, image_width, 3))

    # Iterate through each pixel in the image
    for x in range(image_width):
        for y in range(image_height):
            # Get the camera ray for the current pixel
            camera_ray = camera_rays[x, y, :]

            # Get the shadow ray for the current pixel
            shadow_ray = shadow_rays[x, y, :]

            # Calculate the color for the current pixel
            color = render_equation(camera_ray, shadow_ray, scene, depth)

            # Add the color to the image buffer
            image_buffer[y, x, :] = color

    return image_buffer

def render_preview_image(self, camera_position, camera_direction, camera_fov, image_width, image_height, scene, depth):
    """Renders a preview image using the Pythtracer.

    Args:
        camera_position (np.array): The position of the camera in 3D space.
        camera_direction (np.array): The direction the camera is pointing in 3D space.
        camera_fov (float): The field of view of the camera.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        scene (list): A list of all the objects in the scene.
        depth (int): The maximum recursion depth for the raytracer.

    Returns:
        np.array: A 3D array of RGB values for each pixel in the image.
    """

    # Generate camera rays
    camera_rays = generate_camera_ray(camera_position, camera_direction, camera_fov, image_width, image_height)

    # Generate shadow rays
    shadow_rays = generate_shadow_ray(camera_rays, scene)

    # Generate image buffer
    image_buffer = np.zeros((image_height, image_width, 3))

    # Iterate through each pixel in the image
    for x in range(image_width):
        for y in range(image_height):
            # Get the camera ray for the current pixel
            camera_ray = camera_rays[x, y, :]

            # Get the shadow ray for the current pixel
            shadow_ray = shadow_rays[x, y, :]

            # Calculate the color for the current pixel
            color = recursive_tracing(camera_ray, scene, depth)

            # Add the color to the image buffer
            image_buffer[y, x, :] = color

    return image_buffer

def save_render_image(self, image_buffer, file_name):
    """Saves the rendered image to a file.

    Args:
        image_buffer (np.array): A 3D array of RGB values for each pixel in the image.
        file_name (string): The name of the file to save the image to.
    """

    # Save the image to a file
    np.save(file_name, image_buffer)