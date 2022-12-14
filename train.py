import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

from functools import partial

def train_nn(data: dict) -> None:


    scaler = StandardScaler()
    scaler.fit(data["X_train"])

    X_train_scaled = scaler.transform(data["X_train"])
    X_test_scaled = scaler.transform(data["X_test"])

    n_classes = len(np.unique(data["y_train"]))


    RegularizedDense = partial(
        keras.layers.Dense,
        activation='selu',
        kernel_initializer="lecun_normal",
        kernel_regularizer=keras.regularizers.l1(1e-5))

    model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=(data["X_train"].shape[1], ), name='input'),  # activation='relu',
        # keras.layers.BatchNormalization(),
        #RegularizedDense(64),
        # keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'), #, kernel_regularizer=keras.regularizers.l1(1e-3)),
        # RegularizedDense(500),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(500, activation='relu'), #, kernel_regularizer=keras.regularizers.l1(1e-3)),
        # RegularizedDense(900),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(900, activation='relu'), # , kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l1(1e-3)),
        # RegularizedDense(128),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(128, activation='relu'), # , kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l1(1e-3)),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        # keras.layers.BatchNormalization(),
        keras.layers.Dense(8, activation='relu'),
        # keras.layers.BatchNormalization(),
        #RegularizedDense(8),
        keras.layers.Dense(n_classes, activation='softmax', name='output')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=2e-2),
        metrics=["sparse_categorical_accuracy"]
    )

    print(model.summary())

    # ys = data['y_train'].to_numpy().ravel()
    # weights = compute_class_weight(class_weight='balanced', classes=data['labels'], y=ys)
    # weights[0] = 1
    # weights[-2] = 5
    # weights[-1] = 5
    # weights = dict(enumerate(weights))

    history = model.fit(
        X_train_scaled,
        data['y_train'], 
        epochs=80,
        batch_size=16,
        #class_weight=weights,
        workers=3,
        validation_split=.15,
        verbose=1
    )

    info = pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.savefig("metrics/loss_dec12.png")
    plt.xlabel('Epochs')
    plt.title('Learning Curves')
    # plt.show()

    preds = np.argmax(model.predict(X_test_scaled), axis=-1)# return argmax
    
    print(classification_report(data['y_test'], preds))

    cm = confusion_matrix(data['y_test'], preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data['classes'])

    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    if not os.path.exists(r"models/"):
        os.mkdir(r"models/")
    model.save(r"models/keras_classifier_december12.h5")

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

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GridSearchCV(
            SVC(),
            param_grid={
                'C': np.arange(40,42, 1),
                'gamma': np.linspace(1e-2, 3, 10),
                'kernel': ['rbf'], # 'linear', 
                'random_state': [s]
            },
            scoring='f1_weighted',
            verbose=3,
            cv=5,
            n_jobs=-1,
            refit=True))
    ])
    
    
    pipe.fit(data["X_train"], training_labels)
    print("best parameters: ", pipe['clf'].best_params_)
    preds = pipe.predict(data["X_train"])
    rpt = classification_report(training_labels, preds, zero_division=1)

    print("Training Dataset Report")
    print(rpt)
    
    del preds, rpt
    
    if not os.path.exists(r"models/"):
        os.mkdir(r"models/")
    joblib.dump(pipe, 'models/svc_classifier.compressed', compress=True)


    preds = pipe.predict(data["X_test"])
    rpt = classification_report(test_labels, preds, zero_division=1)
    
    print("Test Dataset Report")
    print(rpt)

    ConfusionMatrixDisplay.from_estimator(
        pipe,
        data["X_test"],
        test_labels,
        display_labels=["Matrix", "Austinite", "Mart. - Aust.","Precipitate", "Defect"],
        cmap="Blues"
        
    )
    # plt.title("SVC - Confusion Matrix.")
    # plt.yticks(rotation=45)
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("metrics/confusion-matrix.png")
    plt.close()


if __name__=="__main__":
    # Train classifier
    from viz import PHASE_MAP
    labels_classes = sorted(PHASE_MAP.items(), key= lambda x: x[0])

    if not os.path.exists("models/"):
        os.mkdir("models/")
    
    if not os.path.exists("metrics/"):
        os.mkdir("metrics/")
    
    dataset = {
        "X_train": pd.read_csv("data/X_train.csv"), 
        "y_train": pd.read_csv("data/y_train.csv"),
        "X_test": pd.read_csv("data/X_test.csv"), 
        "y_test": pd.read_csv("data/y_test.csv"),
        'classes': np.array([l[1] for l in labels_classes])
    }

    dataset['classes'][2] = "Mart./ Aust."

    # For training and testing at the phase level, we can drop the 'picture' column
    # See evaluate.py to compare total picture phase fraction predicitons.
    dataset['X_train'].drop(columns=['picture'], inplace=True)
    dataset['X_test'].drop(columns=['picture'], inplace=True)


    # train_nn(dataset)
    train(dataset)
    # model = keras.models.load_model("models/keras_classifier-v1.h5")
    # print(model.summary())
