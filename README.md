# Bag-of-Words Scene Classifier

Image classification on the 15-scene dataset (Lazebnik et al., 2006) using the Bag-of-Visual-Words approach.

## Pipeline

1. Extract SIFT descriptors from each image
2. Build a visual vocabulary with k-means clustering
3. Encode each image as a normalized histogram of visual words
4. Classify with a 1-nearest-neighbor baseline and a linear SVM

## Structure

```
├── bow_classifier.ipynb   # main notebook
├── src/
│   ├── features.py        # SIFT extraction
│   ├── vocabulary.py      # k-means vocabulary and BoW encoding
│   ├── classifier.py      # 1-NN and SVM classifiers
│   └── utils.py           # dataset loading and plots
├── models/                # saved model files (.pkl)
├── outputs/               # generated figures (accuracy plot, confusion matrices)
└── dataset/
    ├── train/
    │   ├── Bedroom/
    │   └── ...
    └── test/
        ├── Bedroom/
        └── ...
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## How to run

Open `bow_classifier.ipynb` in VS Code and run all cells from top to bottom. The notebook selects the best vocabulary size k on a validation set and reports the final accuracy on the test set. Figures are saved automatically to the `outputs/` folder.