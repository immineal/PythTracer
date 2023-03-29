# Camera Exposure
import numpy as np

def generate_camera_exposure(image):
    """Calculates the exposure of an image.

    Args:
        image (np.array): The image to calculate the exposure for.

    Returns:
        float: The exposure for the image.
    """

    exposure = np.mean(image)
    return exposure

# Depth of Field

def generate_depth_of_field(focal_length, aperture, object_distance, image_distance):
    """Calculates the depth of field for a given focal length, aperture, object distance, and image distance.

    Args:
        focal_length (float): The focal length of the camera.
        aperture (float): The aperture of the camera.
        object_distance (float): The distance of the object from the camera.
        image_distance (float): The distance of the image from the camera.

    Returns:
        float: The depth of field for the given parameters.
    """

    depth_of_field = (aperture * (object_distance - image_distance)) / focal_length
    return depth_of_field

# Motion Blur

def generate_motion_blur(image, exposure_time):
    """Calculates the motion blur of an image.

    Args:
        image (np.array): The image to calculate the motion blur for.
        exposure_time (float): The exposure time of the image.

    Returns:
        np.array: The motion blurred image.
    """

    motion_blur = np.multiply(image, exposure_time)
    return motion_blur