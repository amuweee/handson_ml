"""
Ensembling is a aggregator of multiple different predictors
    A prediction is returned based on the class that gets the most votes overall
        This is called Hard Voting CLassifer
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np

# create and train a voting classifer composed of three diverse classifiers
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data  # petal length and width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


voting_clf = VotingClassifier(
    estimators=[("lr", log_clf), ("rf", rnd_clf), ("scv", svm_clf)], voting="hard"
)
voting_clf.fit(X_train, y_train)

# to view the performance of each classifier
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# if classifers can estimate probabilities, ie have a predict_proba() method
#   then can predict the class with the highest class probability
#   this is called soft voting


"""
Bagging and Pasting
    instead of using diverse set of classifiers, use the same classifier on random subsets of the training set
    the prediction is based on the statistical mode (most common class)

"""

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

print(y_pred)

# by default, not all samples are tested in bagging (bootstrap=True)
# set oob_score=True to request an automatic oob evalutation after training.

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=500,
    bootstrap=True,
    n_jobs=-1,
    oob_score=True,
)
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)

# see if this accuracy actually comes through
from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
