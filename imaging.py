# Gamma Correction
import cv2
import numpy as np

def gamma_correction(image, gamma):
    """Applies gamma correction to the image.

    Args:
        image (np.array): The image to be corrected.
        gamma (float): The gamma correction value.

    Returns:
        np.array: The corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Tone Mapping

def tone_mapping(image, gamma, exposure):
    """Applies tone mapping to the image.

    Args:
        image (np.array): The image to be mapped.
        gamma (float): The gamma correction value.
        exposure (float): The exposure value.

    Returns:
        np.array: The mapped image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * (exposure * 255) for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Color Correction

def color_correction(image, gain, bias):
    """Applies color correction to the image.

    Args:
        image (np.array): The image to be corrected.
        gain (float): The gain value.
        bias (float): The bias value.

    Returns:
        np.array: The corrected image.
    """
    table = np.array([((i / 255.0) * gain + bias) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Film Grain

def film_grain(image, amount):
    """Applies film grain to the image.

    Args:
        image (np.array): The image to be grainy.
        amount (float): The amount of grain to add.

    Returns:
        np.array: The grainy image.
    """
    grain = np.random.randint(0, 255, size=image.shape).astype("uint8")
    return cv2.addWeighted(image, 1.0, grain, amount, 0.0)