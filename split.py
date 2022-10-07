import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # split data into training and test data for supervised models.

    FEATURES = ["area", "area_filled", "axis_major_length", "axis_minor_length",
                "eccentricity", "equivalent_diameter_area", "feret_diameter_max",
                "intensity_max", "intensity_mean", "intensity_min", "perimeter",
                "solidity"]

    LABEL = ["phase_number"]

    SEED = 42
    
    data = pd.read_csv(r"data/morphological_data.csv", index_col=False)
    cols_to_keep = FEATURES + LABEL
    data.drop(columns=[c for c in data.columns if c not in cols_to_keep], inplace=True)
    
    # Initial train test split
    X_train, X_test, y_train, y_test = \
        train_test_split(data[FEATURES], data[LABEL], random_state=SEED)
    
    # Create validation data set
    X_valid, X_test, y_valid, y_test = \
        train_test_split(X_test[FEATURES], y_test[LABEL], random_state=SEED)

    if not os.path.exists("data"):
        os.mkdir("data/")

    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    X_valid.to_csv("data/X_valid.csv", index=False)
    y_valid.to_csv("data/y_valid.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
