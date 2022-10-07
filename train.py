import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

def train(features: pd.DataFrame, labels: pd.DataFrame, s: int=2) -> None:
    # Train a classification model to predict phases in AM steels.

    # Baseline Dummy Classifier
    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=s)
    dummy_clf.fit(features, labels)

    dummy_preds = dummy_clf.predict(features)

    print(dummy_clf.score(features, labels))
    print(classification_report(labels, dummy_preds))


    # Machine Learning Classifier.
    pipe = Pipeline(
        [('scaler', StandardScaler()), 
         ('clf', SVC(max_iter=100000, random_state=s))]
    )

    pipe.fit(features, labels)

    preds = pipe.predict(features)
    print(pipe.score(features, labels))
    print(classification_report(labels, preds, zero_division=1))


if __name__=="__main__":
    # Train classifier

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")

    if not os.path.exists("models/"):
        os.mkdir("models/")
    
    if not os.path.exists("metrics/"):
        os.mkdir("metrics/")

    print(type(X_train), type(y_train))
 
    train(X_train, y_train)
