from core import generate_camera_ray


from gui import render_preview_image
from gui import save_render_image

def render_example_scene(camera_position, camera_direction, camera_fov, image_width, image_height, scene, depth):
    """Renders an example scene.

    Args:
        camera_position (np.array): The position of the camera in 3D space.
        camera_direction (np.array): The direction the camera is pointing in 3D space.
        camera_fov (float): The field of view of the camera.
        image_width (int): The width of the image in pixels.
        image_height (int): The height of the image in pixels.
        scene (np.array): A 3D array of scene objects.
        depth (int): The maximum depth of the recursive tracing.

    Returns:
        np.array: A 3D array of the rendered image.
    """

    # Generate camera rays
    camera_ray = generate_camera_ray(camera_position, camera_direction, camera_fov, image_width, image_height)

    # Create the render window
    render_window = RenderWindow()

    # Create the render preview image
    render_preview_image = render_preview_image(camera_position, camera_direction, camera_fov, image_width, image_height, scene, depth)

    # Render the image
    image_buffer = render_window.render_image(camera_position, camera_direction, camera_fov, image_width, image_height, scene, depth)

    # Update the progress bar
    render_progress_bar.update_progress(image_buffer)

    # Save the render image
    save_render_image(image_buffer, "example_scene.png")

    return image_buffer