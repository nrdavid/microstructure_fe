import os
import json
from textwrap import indent
import numpy as np
import sqlite3
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

def measured_area_fraction() -> pd.DataFrame:

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
        micrograph"""
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

    AllData = pd.read_csv(r"data/morphological_data.csv", index_col=False)
    cols_to_keep = FEATURES + LABEL + IMAGE
    AllData.drop(columns=[c for c in AllData.columns if c not in cols_to_keep], inplace=True)

    Model = joblib.load(r"models/classifier.compressed")

    X = AllData.copy(deep=True)
    X.drop(columns=['phase_number'], inplace=True)

    MeasuredArea = measured_area_fraction()

    values, loss = evaluate(Model, X, MeasuredArea, PHASE_MAP)

    with open("metrics/phase-area-rmse.json", "w") as f:
        f.write(json.dumps(loss, indent=2))

    plot_predicted_versus_true(values, 'metrics')

    