"""SIFT descriptor extraction."""

import cv2
import numpy as np


def extract_descriptors(paths):
    """Extract SIFT descriptors from a list of image paths.
    
    Args:
        paths : list of image path strings
    
    Returns:
        descriptors : list of numpy arrays, one per image, shape (N_i, 128).
                      Empty array (0, 128) if no keypoints are found.
    """
    sift = cv2.SIFT_create()
    descriptors = []

    for i, path in enumerate(paths):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: could not read {path}")
            descriptors.append(np.empty((0, 128), dtype=np.float32))
            continue

        _, desc = sift.detectAndCompute(image, None)
        if desc is None:
            descriptors.append(np.empty((0, 128), dtype=np.float32))
        else:
            descriptors.append(desc.astype(np.float32))

        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  {i + 1}/{len(paths)} images processed")

    return descriptors