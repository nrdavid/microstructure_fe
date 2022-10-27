import os
import numpy as np
import pandas as pd
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import joblib
from matplotlib import pyplot as plt

from tensorflow import keras


def train_nn(data: dict) -> None:


    scaler = StandardScaler()
    scaler.fit(data["X_train"])

    X_train_scaled = scaler.transform(data["X_train"])
    X_test_scaled = scaler.transform(data["X_test"])

    n_classes = len(np.unique(data["y_train"]))

    model = keras.models.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(data["X_train"].shape[1], ), name='input'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(900, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(8, activation='sigmoid'),
        keras.layers.Dense(n_classes, activation='softmax', name='output')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=1e-2),
        metrics=['accuracy']
    )

    # ys = data['y_train'].to_numpy().ravel()
    # weights = compute_class_weight(class_weight='balanced', classes=data['labels'], y=ys)
    # weights[0] = 1
    # weights[-2] = 5
    # weights[-1] = 5
    # weights = dict(enumerate(weights))

    history = model.fit(
        X_train_scaled,
        data['y_train'], 
        epochs=36,
        batch_size=16,
        #class_weight=weights,
        workers=3,
        validation_split=.15,
        verbose=1
    )

    info = pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.savefig("metrics/loss.png")
    # plt.show()

    preds = np.argmax(model.predict(X_test_scaled), axis=-1)# return argmax
    
    print(classification_report(data['y_test'], preds))


    cm = confusion_matrix(data['y_test'], preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data['classes'])

    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return None

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
         ('clf', GridSearchCV(
            SVC(),
            param_grid={
                'C': np.arange(40, 50, 1),
                'gamma': np.arange(1e-2, 3, 10),
                'kernel': ['rbf'],
                'random_state': [s]
            },
            scoring='f1_weighted',
            verbose=2,
            cv=5,
            n_jobs=-1,
            refit=True
         ))]
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

    # For training and testing at the phase level, we can drop the 'picture' column
    # See evaluate.py to compare total picture phase fraction predicitons.
    dataset['X_train'].drop(columns=['picture'], inplace=True)
    dataset['X_test'].drop(columns=['picture'], inplace=True)


    train_nn(dataset)
