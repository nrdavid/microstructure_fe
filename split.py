import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
        "area",
        "area_bbox",
        "area_filled",
        "area_convex",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "extent",
        "feret_diameter_max",
        "intensity_max",
        "intensity_mean",
        "intensity_min",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity"
]

LABEL = ["phase_number"]
IMAGE = ["picture"]

SEED = 42

def train_test_images(images: list, train_size: float=0.8 , seed: int=SEED) -> dict:
    np.random.seed(seed=seed)

    num_pics = len(images)
    train_size = int(np.floor(num_pics * train_size))
    test_size = num_pics - train_size

    split = {
        "train_pics": [],
        "test_pics": []
    }
    # should update to ensure train samples contain all phases.
    for _ in range(num_pics):
        sample = np.random.choice(images) # not sure why replace=False is not working
        images.remove(sample) # just remove picture from list instead.

        if len(split['train_pics']) < train_size:
            split['train_pics'].append(sample)
        elif len(split['test_pics']) < test_size:
            split['test_pics'].append(sample)
        else:
            break

    return split

def partition(D: pd.DataFrame, splits: dict) -> tuple:

    X_train = D[D["picture"].isin(splits['train_pics'])].sample(frac=1)
    y_train = X_train["phase_number"]
    X_train.drop(columns=['phase_number'], inplace=True)

    X_test = D[D["picture"].isin(splits['test_pics'])].sample(frac=1)
    y_test = X_test["phase_number"]
    X_test.drop(columns=['phase_number'], inplace=True)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # split data into training and test data for supervised models.

    data = pd.read_csv(r"data/morphological_data.csv", index_col=False)
    cols_to_keep = FEATURES + LABEL + IMAGE
    data.drop(columns=[c for c in data.columns if c not in cols_to_keep], inplace=True)

    images = list(data['picture'].unique())
    X_train, X_test, y_train, y_test = partition(data, train_test_images(images))

    
    # # Initial train test split prior to tracking image.
    # X_train, X_test, y_train, y_test = \
    #     train_test_split(data[FEATURES], data[LABEL], random_state=SEED)
    
    if not os.path.exists("data"):
        os.mkdir("data/")

    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)

    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    