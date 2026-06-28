"""Visual vocabulary construction and bag-of-words encoding."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def build_vocabulary(descriptors, k, random_seed=42):
    """Build a visual vocabulary of k words using k-means.
    
    Args:
        descriptors : list of numpy arrays (one per training image)
        k           : number of visual words (vocabulary size)
        random_seed : for reproducibility
    
    Returns:
        vocabulary : fitted MiniBatchKMeans object with k cluster centers
    """
    # Concatenate all descriptors from all training images
    all_descriptors = np.vstack([d for d in descriptors if len(d) > 0])
    print(f"  Total descriptors for clustering: {len(all_descriptors)}")

    vocabulary = MiniBatchKMeans(n_clusters=k, random_state=random_seed, n_init=3)
    vocabulary.fit(all_descriptors)
    return vocabulary


def encode_bow(descriptors, vocabulary):
    """Encode each image as a normalized bag-of-words histogram.
    
    Each descriptor is assigned to the nearest visual word.
    The histogram is L1-normalized to get relative frequencies.
    
    Args:
        descriptors : list of numpy arrays (one per image)
        vocabulary  : fitted MiniBatchKMeans object
    
    Returns:
        features : numpy array of shape (n_images, k)
    """
    k = vocabulary.n_clusters
    features = np.zeros((len(descriptors), k), dtype=np.float32)

    for i, desc in enumerate(descriptors):
        if len(desc) == 0:
            continue
        # Assign each descriptor to the nearest visual word
        word_assignments = vocabulary.predict(desc)
        # Count occurrences of each word
        histogram = np.bincount(word_assignments, minlength=k).astype(np.float32)
        # L1 normalization: convert counts to relative frequencies
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        features[i] = histogram

    return features