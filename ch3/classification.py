# fetch data set
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1)
mnist.keys()


# Look at the data arrays
X, y = mnist["data"], mnist["target"]
X.shape
y.shape


# Lets visualize it
import matplotlib as mpl
import matplotlib.pyplot as plt

rownum = 0

some_digit = X[rownum]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

y[rownum]

# turn y from str to int
import numpy as np

y = y.astype(np.uint8)


# split the data to train and test (first 60k are train, last 10k are test)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# Try something simple: create a classifier that can Identify "5"
y_train_5 = y_train == 5
y_test_5 = y_test == 5

# Pick a classifier
# SGD = Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])  # yay it got correct


# Measuring accuracy using Cross-Validation
# Use K-fold cross-validation with three folds

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Let's set the baseline for comparison
from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


# Introducing: Confusion Matrix

# Let's first create a prediction set
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)


# Precision and recall:
#   Precision: TP / (TP+FP)
#   Recall:    TP / (TP+FN)

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

# F1 score -> harmonic mean of precision and recall
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)


# look at the decision threashold
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = y_scores > threshold


# how to decide what threashold to set?
# use cross_val_predict to reutrn threashold of each fold

y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)


# Let's plot the precsion and recall curve to visualize thresholds

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
