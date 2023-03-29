# Multiprocessing
import multiprocessing
import numpy as np
import threading
from core import recursive_tracing, generate_camera_ray

def multiprocessing_raytracer(rays, scene, depth):
    """Uses multiprocessing to speed up ray tracing.

    Args:
        rays (np.array): An array of rays.
        scene (list): A list of objects in the scene.
        depth (int): The maximum depth of recursion.

    Returns:
        np.array: An array of colors for each ray.
    """
    with multiprocessing.Pool() as pool:
        results = pool.starmap(recursive_tracing, [(rays[i], scene, depth) for i in range(rays.shape[0])])
    return np.array(results)
    
# Anti Aliasing

def anti_aliasing(rays, scene, depth):
    """Uses anti-aliasing to reduce aliasing artifacts.

    Args:
        rays (np.array): An array of rays.
        scene (list): A list of objects in the scene.
        depth (int): The maximum depth of recursion.

    Returns:
        np.array: An array of colors for each ray.
    """
    num_samples = 4
    sample_rays = np.array([
        rays + np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]])
    ])
    results = np.array([recursive_tracing(sample_rays[i], scene, depth) for i in range(num_samples)])
    return np.mean(results, axis=0)
    
# Threading

def threading_raytracer(rays, scene, depth):
    """Uses threading to speed up ray tracing.

    Args:
        rays (np.array): An array of rays.
        scene (list): A list of objects in the scene.
        depth (int): The maximum depth of recursion.

    Returns:
        np.array: An array of colors for each ray.
    """
    threads = []
    results = np.empty(shape=(rays.shape[0], 3))
    for i in range(rays.shape[0]):
        t = threading.Thread(target=recursive_tracing, args=(rays[i], scene, depth, results[i]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return results
    
# Adaptive Sampling

def adaptive_sampling(image, threshold):
    """Uses adaptive sampling to reduce the number of rays used for rendering.

    Args:
        image (np.array): The image to be rendered.
        threshold (float): The threshold for the adaptive sampling.

    Returns:
        np.array: An array of rays for the image.
    """
    rays = np.empty(shape=(image.shape[0] * image.shape[1], 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.max(image[i, j] - image[i-1, j-1]) > threshold:
                rays[i*image.shape[1] + j] = generate_camera_ray(camera_position, camera_direction, camera_fov, image_width, image_height)
    return rays