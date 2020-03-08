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


# ROC - Receuver Operating Characteristics curve
# ROC curve plots true positive against true negative
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")


plot_roc_curve(fpr, tpr)
plt.show()
# The area under the curve is the model performance. More away to the dotted line = Better

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)


# Let's compare the SGDClasifier to RandomForest

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

y_scores_forest = y_probas_forest[:, 1]  # scores = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# Let's plot
plt.plot(fpr, tpr, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

# RandomForestClassifier is better


###______ MULTICLASS CLASSIFICCATION
# One versus the rest
# One versus one
# some algorithms in sklearn will select OvR or OvO automatically

from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_clf.predict([some_digit])

# see prediction per class
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores

# To specify OvR or OvO classifier:
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)


# Get some scoring on cross_val
cross_val_predict(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# scale the data to improve accuracy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# Error analysis
# Time to compare to other models and look at way to improve predictions
# Get confusion matrix
y_train_pred = cross_val_predict(
    sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"
)

conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# too many values, plot htem in a graph
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# turn the confusion matrix to show by % of errors instead of absolute numbers
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()  # rows=actual classes, col=predicted classes


# Multilabel classification: multiple predictions (list) per label
from sklearn.neighbors import KNeighborsClassifier

y_train_large = y_train >= 7  # labels that are 7,8 or 9
y_train_odd = y_train % 2 == 1
y_multilabel = np.c_[
    y_train_large, y_train_odd
]  # concat into a list of results on whether the digit is [Large, Odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

# compute F1 score across all labels
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

