"""Visual vocabulary construction and bag-of-words encoding."""

import numpy as np
from sklearn.cluster import KMeans

N_SAMPLE = 100_000  # number of descriptors sampled for k-means (as per assignment)


def build_vocabulary(descriptors, k, random_seed=42):
    """Build a visual vocabulary of k words using k-means clustering.

    A random subset of N_SAMPLE descriptors is drawn from all training
    descriptors before clustering, following the assignment guidelines.

    Args:
        descriptors : list of numpy arrays (one per training image)
        k           : number of visual words (vocabulary size)
        random_seed : for reproducibility

    Returns:
        vocabulary : fitted KMeans object with k cluster centers
    """
    all_descriptors = np.vstack([d for d in descriptors if len(d) > 0])
    print(f"  Total descriptors extracted : {len(all_descriptors)}")

    # Sample at most N_SAMPLE descriptors for clustering
    if len(all_descriptors) > N_SAMPLE:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(all_descriptors), N_SAMPLE, replace=False)
        sample = all_descriptors[idx]
    else:
        sample = all_descriptors
    print(f"  Descriptors used for k-means: {len(sample)}")

    vocabulary = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    vocabulary.fit(sample)
    return vocabulary


def encode_bow(descriptors, vocabulary):
    """Encode each image as a normalized bag-of-words histogram.

    Each descriptor is assigned to the nearest visual word (hard assignment).
    The histogram is L1-normalized to obtain relative frequencies.

    Args:
        descriptors : list of numpy arrays (one per image)
        vocabulary  : fitted KMeans object

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
        # L1 normalisation: convert counts to relative frequencies
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        features[i] = histogram

    return features