"""Image classifiers: 1-nearest-neighbor baseline and linear SVM."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def train_knn(X_train, y_train):
    """Train a 1-nearest-neighbor classifier."""
    model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, C=1.0, random_seed=42):
    """Train a one-vs-rest linear SVM classifier."""
    model = LinearSVC(C=C, max_iter=5000, random_state=random_seed)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Return predictions and accuracy on a given set."""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return predictions, acc