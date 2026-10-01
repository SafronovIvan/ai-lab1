import numpy as np
from sklearn import linear_model
from utilities import visualize_classifier

X = np.array([
    [3.1, 7.2], [4, 6.7], [2.9, 8],
    [5.1, 4.5], [6, 5], [5.6, 5],
    [3.3, 0.4], [3.9, 0.9], [2.8, 1],
    [0.5, 3.4], [1, 4], [0.6, 4.9]
])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# === C = 1 ===
classifier = linear_model.LogisticRegression(solver='lbfgs', C=1, max_iter=1000)
classifier.fit(X, y)
visualize_classifier(classifier, X, y)
print("C=1 done, close the plot to see C=100...")

# === C = 100 ===
classifier_100 = linear_model.LogisticRegression(solver='lbfgs', C=100, max_iter=1000)
classifier_100.fit(X, y)
visualize_classifier(classifier_100, X, y)
print("C=100 done.")