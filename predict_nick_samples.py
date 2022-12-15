import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import numpy as np
import sqlite3 
import re


def fix_units(df):

    def get_pixel_to_area_conversion(im: str, db:str=r"/home/shenron/Documents/School/MSE/MSE 454/Project/microstructure_fe/MetalDAM/MetalDAM_metadata.sql") -> float:
        with sqlite3.connect(db) as conn:
            cur = conn.cursor()

            sql = f"""SELECT pixels_per_micron FROM micrograph WHERE image_path LIKE '%{im}.jpg';"""
            cur.execute(sql)
            conversion = cur.fetchone()
            conversion = float(conversion[0])
        return conversion

    fixed_units = []
    for i in df.picture.unique():
        conversion = get_pixel_to_area_conversion(i)
        area_cols = [c for c in df.columns if re.search(r"area|diameter|length|perimeter",c)]
        df_temp = df[df['picture']==i]
        for col in area_cols:
            if col.__contains__('area') and (col != "equivalent_diameter_area"):
                df_temp[col] = df_temp[col] / conversion**2
            else:
                df_temp[col] = df_temp[col] / conversion
        fixed_units.append(df_temp)

    return pd.concat(fixed_units)
    

if __name__ == "__main__":
    import warnings
    from evaluate import measured_area_fraction
    from viz import PHASE_MAP
    warnings.filterwarnings('ignore')
    print(PHASE_MAP)
    PHASE_MAP[2] = 'Martensite_Austenite'
    
    Model = keras.models.load_model(r"models/keras_classifier_december13.h5")
    print(Model)

    dataset = {
        "X_train": pd.read_csv("data/X_train.csv"),
        "y_train": pd.read_csv("data/y_train.csv"),
        "X_test": pd.read_csv("data/X_test.csv"),
        "y_test": pd.read_csv("data/y_test.csv")
    }

    test_images = dataset['X_test'].picture.unique()

    cols = [c for c in dataset["X_train"].columns if c != 'picture']

    scaler = StandardScaler()
    scaler.fit(dataset["X_train"][cols])

    df_nick_samples = pd.read_pickle(r"misc/tom_final.pkl")
    df_nick_samples['picture'] = df_nick_samples['picture'].map(lambda x: x.split('.')[0])

    df_nick_samples = df_nick_samples[df_nick_samples['picture'].isin(test_images)]
    print(df_nick_samples)

    """Fixing Units"""
    df_nick_samples = fix_units(df_nick_samples)

    X_nick_scaled = scaler.transform(df_nick_samples[cols])
   
    nick_preds = np.argmax(Model.predict(X_nick_scaled), axis=-1).reshape(-1,1)

    df_nick_samples['predictions'] = nick_preds
    df_nick_samples['phase'] = df_nick_samples['predictions'].map(lambda x: PHASE_MAP[x])
    # 'area_filled',
    phase_area_per_image = df_nick_samples[['area', 'phase', 'picture']].\
                                    groupby(by=['picture', 'phase']).\
                                    sum().\
                                    reset_index().\
                                    rename(columns={'area':'Predicted Area'})

    MeasuredArea_test = measured_area_fraction(df_nick_samples['picture'].unique())
    # print(MeasuredArea_test)
    print("----------")

    x = phase_area_per_image[['picture', 'phase', 'Predicted Area']].pivot(index='picture', columns='phase', values='Predicted Area')
    
    x.rename(columns={'Austinite': 'Austenite'}, inplace=True)
    x['Matrix'] = 0
    x['Precipitate'] = 0
    x['Defect'] = 0

    x.to_pickle('misc/Nick_Preds_units_fixed.pkl')

    Nick_Test = pd.merge(
        MeasuredArea_test, 
        phase_area_per_image,
        how='left', 
        left_on=['picture', 'phase'],
        right_on=['picture', 'phase']
    ).fillna(0)

    Nick_Test.rename(columns={'area':'True Area'}, inplace=True)
    print(Nick_Test)
    Nick_Test.to_pickle("metrics/Nick_prediction_results.pkl")
    
    func = lambda d: round(np.sqrt(mean_squared_error(d['True Area'], d['Predicted Area'])),2)
    df_new = Nick_Test.groupby(by=['phase']).apply(func).reset_index()
    print(df_new)
    df_new.to_pickle("metrics/Nick_prediction_RMSE.pkl")

    