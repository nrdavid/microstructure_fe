import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import joblib
from matplotlib import pyplot as plt

def train(data: dict, s: int=2) -> Pipeline:

    # Train a classification model to predict phases in AM steels.

    training_labels = data["y_train"].to_numpy().ravel()
    test_labels = data["y_test"].to_numpy().ravel()
    
    # store confusion matrices for training and test set.
    
    # # Baseline Dummy Classifier
    # dummy_clf = DummyClassifier(strategy="most_frequent", random_state=s)
    # dummy_clf.fit(features, labels)

    # dummy_preds = dummy_clf.predict(features)

    # print(dummy_clf.score(features, labels))
    # print(classification_report(labels, dummy_preds))


    # Machine Learning Classifier.
    pipe = Pipeline(
        [('scaler', StandardScaler()), # convert features into z-score.
         ('clf', SVC(max_iter=100000, random_state=s))]
    )
    
    pipe.fit(data["X_train"], training_labels)

    preds = pipe.predict(data["X_train"])
    rpt = classification_report(training_labels, preds, zero_division=1)

    print("Training Dataset Report")
    print(rpt)
    
    del preds, rpt
    
    if not os.path.exists(r"models/"):
        os.mkdir(r"models/")
    joblib.dump(pipe, 'models/classifier.compressed', compress=True)

    preds = pipe.predict(data["X_test"])
    rpt = classification_report(test_labels, preds, zero_division=1)
    
    print("Test Dataset Report")
    print(rpt)

    ConfusionMatrixDisplay.from_estimator(
        pipe,
        data["X_test"],
        test_labels,
        display_labels=["Matrix", "Austinite", "Mart. - Aust.","Precipitate", "Defect"]
        
    )
    plt.title("SVC - Confusion Matrix.")
    plt.yticks(rotation=45)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("metrics/confusion-matrix.png")
    plt.close()


if __name__=="__main__":
    # Train classifier

    if not os.path.exists("models/"):
        os.mkdir("models/")
    
    if not os.path.exists("metrics/"):
        os.mkdir("metrics/")
    
    dataset = {
        "X_train": pd.read_csv("data/X_train.csv"), 
        "y_train": pd.read_csv("data/y_train.csv"),
        "X_test": pd.read_csv("data/X_test.csv"), 
        "y_test": pd.read_csv("data/y_test.csv")
    }

    train(dataset)
