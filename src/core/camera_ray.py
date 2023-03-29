import numpy as np

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