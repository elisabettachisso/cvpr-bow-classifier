"""Utility functions for loading images and plotting results."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(root_path):
    """Load image paths and labels from a folder structured as root/category/image.jpg.
    
    Returns:
        paths  : list of image path strings
        labels : numpy array of integer labels
        categories : list of category names (sorted)
    """
    categories = sorted(os.listdir(root_path))
    paths = []
    labels = []

    for label, category in enumerate(categories):
        folder = os.path.join(root_path, category)
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                paths.append(os.path.join(folder, filename))
                labels.append(label)

    return paths, np.array(labels), categories


def plot_accuracy_vs_k(k_values, knn_accuracies, svm_accuracies):
    """Plot validation accuracy as a function of vocabulary size k."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, knn_accuracies, 'o--', label='1-NN')
    ax.plot(k_values, svm_accuracies, 'o-', label='Linear SVM')
    ax.set_xlabel('Vocabulary size (k)')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_title('Accuracy vs vocabulary size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'accuracy_vs_k.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, categories, title):
    """Plot a normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    fig.savefig(OUTPUT_DIR / f'confusion_matrix_{safe_title}.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)