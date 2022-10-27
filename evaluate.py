import os
import json
import numpy as np
import sqlite3
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

def measured_area_fraction(pics: np.ndarray) -> pd.DataFrame:

    pic_str = ",".join([f"'images/{p}.jpg'" for p in pics])

    with sqlite3.connect(r"MetalDAM/MetalDAM_metadata.sql") as conn:
        cursor = conn.cursor()
        sql = r"""
        SELECT 
        image_path,
        (label0 / pixels_per_micron) AS Matrix,
        (label1 / pixels_per_micron) AS Austinite,
        (label2 / pixels_per_micron) AS Martensite_Austenite,
        (label3 / pixels_per_micron) AS Precipitate,
        (label4 / pixels_per_micron) AS Defect
        FROM
        micrograph
        WHERE image_path in ({})
        """.format(pic_str)

        cursor.execute(sql)
        cols = ['picture', 'Matrix', 'Austinite', 'Martensite_Austenite', 'Precipitate', 'Defect']
        df = pd.DataFrame((row for row in cursor.fetchall()), columns=cols)
    
    df['picture'] = df['picture'].map(lambda x: re.search(r"micrograph[0-9]{0,3}", x).group())
    df = pd.melt(df, id_vars=['picture'], var_name='phase', value_name='area')
    
    return df

def evaluate(model: Pipeline, X: pd.DataFrame, measured: pd.DataFrame, key: dict) -> tuple:
    
    X_temp = X.copy(deep=True)
    X_temp.drop(columns=['picture'], inplace=True)

    X['phase'] = model.predict(X_temp)
    
    phase_area_per_image = X[['area', 'area_filled', 'phase', 'picture']].\
                            groupby(by=['picture', 'phase']).\
                            sum().\
                            reset_index()

    phase_area_per_image['phase'] = phase_area_per_image['phase'].map(lambda k: key[k])

    df = pd.merge(
            measured, 
            phase_area_per_image,
            how='left', 
            left_on=['picture', 'phase'],
            right_on=['picture', 'phase'])
        
    df.fillna(0, inplace=True)
    df.rename(columns={'area_x':'True Area', 'area_y':'Predicted Area'}, inplace=True)

    loss_metrics = {}
    for group in df.groupby(by='phase'):
        phase, d = group
        loss_metrics[phase] = round(np.sqrt(mean_squared_error(d['True Area'], d['Predicted Area'])),2)

    return df, loss_metrics

if __name__ == "__main__":
    from split import FEATURES, LABEL, IMAGE
    from viz import PHASE_MAP, plot_predicted_versus_true

    dataset = {
        "X_train": pd.read_csv("data/X_train.csv"), 
        "X_test": pd.read_csv("data/X_test.csv"), 
    }

    train_pics = dataset['X_train']['picture'].unique()
    test_pics = dataset['X_test']['picture'].unique()

    Model = joblib.load(r"models/classifier.compressed")

    MeasuredArea_train = measured_area_fraction(train_pics)
    train_values, train_loss = evaluate(Model, dataset['X_train'], MeasuredArea_train, PHASE_MAP)
    train_values['split'] = 'train'


    MeasuredArea_test = measured_area_fraction(test_pics)
    test_values, test_loss = evaluate(Model, dataset['X_test'], MeasuredArea_test, PHASE_MAP)
    test_values['split'] = 'test'

    results = pd.concat([train_values, test_values])

    errors = {
        'train': train_loss,
        'test': test_loss
    }

    with open("metrics/phase-area-rmse.json", "w") as f:
        f.write(json.dumps(errors, indent=3))

    plot_predicted_versus_true(results, 'metrics')

    