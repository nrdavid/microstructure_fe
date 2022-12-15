import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from matplotlib import pyplot as plt

def plot_predicted_versus_true(data:pd.DataFrame, preds: pd.DataFrame, folder:str ) -> None:

    for group in data.groupby(by='phase'):
        phase, df = group


        train = df[df['split']=='train']
        test = df[df['split']=='test']

        nick_preds = preds[preds['phase']==phase]

        print(train.shape, test.shape)
        if phase == 'Austinite':
            phase = 'Austenite'

        plt.title(f'Residual Error For {phase} Phase', fontsize=14)
        plt.scatter(train['True Area'], train['Predicted Area'], marker='x', c='black', label='Train Images', alpha=0.5)
        plt.scatter(test['True Area'], test['Predicted Area'], marker='P', c='orange', label='Labeld Test Images')
        plt.scatter(nick_preds['True Area'], nick_preds['Predicted Area'], marker='8', c='green', label='Segmented Test Images', alpha=0.7)
        plt.ylabel('Predicted Total Phase Area (microns)', fontsize=14)
        plt.xlabel('True Total Phase Area (microns)', fontsize=14)
        print("phase: " + phase)
        plt.ylim(0, 525 if phase in ('Matrix', 'Austenite') else None)
        plt.xlim(0, 525 if phase in ('Matrix', 'Austenite') else None)
        plt.tight_layout()
        plt.legend(loc='best')
        plt.savefig(f"{folder}/NN-Residuals-{phase} - dec13 - wo micrograph27.png")
        # plt.show()
        plt.clf()


if __name__ == "__main__":
    import warnings
    from evaluate import measured_area_fraction
    from viz import PHASE_MAP
    warnings.filterwarnings('ignore')
    print(PHASE_MAP)
    PHASE_MAP[2] = 'Martensite_Austenite'
    
    Model = keras.models.load_model(r"models/keras_classifier_december6.h5")
    print(Model)

    dataset = {
        "X_train": pd.read_csv("data/X_train.csv"),
        "y_train": pd.read_csv("data/y_train.csv"),
        "X_test": pd.read_csv("data/X_test.csv"),
        "y_test": pd.read_csv("data/y_test.csv")
    }

    cols = [c for c in dataset["X_train"].columns if c != 'picture']

    scaler = StandardScaler()
    scaler.fit(dataset["X_train"][cols])

    X_train_scaled = scaler.transform(dataset["X_train"][cols])
    X_test_scaled = scaler.transform(dataset["X_test"][cols])

    train_preds = np.argmax(Model.predict(X_train_scaled), axis=-1).reshape(-1,1)
    Train_df = pd.merge(dataset['X_train'], dataset['y_train'], how='inner', left_index=True, right_index=True).round(3)
    Train_df['predictions'] = train_preds
    Train_df['phase'] = Train_df['predictions'].map(lambda x: PHASE_MAP[x])
    MeasuredArea_train = measured_area_fraction(Train_df['picture'].unique())

    phase_area_per_image = Train_df[['area', 'area_filled', 'phase', 'picture']].\
                                    groupby(by=['picture', 'phase']).\
                                    sum().\
                                    reset_index().\
                                    rename(columns={'area':'Predicted Area'})

    Train = pd.merge(
        MeasuredArea_train, 
        phase_area_per_image,
        how='left', 
        left_on=['picture', 'phase'],
        right_on=['picture', 'phase']
    ).fillna(0)
    Train.rename(columns={'area':'True Area'}, inplace=True)

    Train['split'] = 'train'
    
    print(Train)
    del phase_area_per_image

    test_preds = np.argmax(Model.predict(X_test_scaled), axis=-1).reshape(-1,1)
    Test_df = pd.merge(dataset['X_test'], dataset['y_test'], how='inner', left_index=True, right_index=True).round(3)
    Test_df['predictions'] = test_preds
    Test_df['phase'] = Test_df['predictions'].map(lambda x: PHASE_MAP[x])
    MeasuredArea_test = measured_area_fraction(Test_df['picture'].unique())

    phase_area_per_image = Test_df[['area', 'area_filled', 'phase', 'picture']].\
                                    groupby(by=['picture', 'phase']).\
                                    sum().\
                                    reset_index().\
                                    rename(columns={'area':'Predicted Area'})

    Test = pd.merge(
        MeasuredArea_test, 
        phase_area_per_image,
        how='left', 
        left_on=['picture', 'phase'],
        right_on=['picture', 'phase']
    ).fillna(0)
    
    Test.rename(columns={'area':'True Area'}, inplace=True)
    Test['split'] = 'test'

    results = pd.concat([Train, Test])
    print(results)

    nick_data = pd.read_pickle(r"metrics/Nick_prediction_results.pkl")

    plot_predicted_versus_true(results, nick_data, 'metrics')

    loss_metrics = {}
    for group in results.groupby(by=['split', 'phase']):
        phase, d = group
        loss_metrics[phase] = round(np.sqrt(mean_squared_error(d['True Area'], d['Predicted Area'])),2)
    
    print(loss_metrics)

    func = lambda d: round(np.sqrt(mean_squared_error(d['True Area'], d['Predicted Area'])),2)
    df_new = results.groupby(by=['split', 'phase']).apply(func).reset_index()

    df_new2 = df_new.pivot(index='phase', columns='split', values=0)
    print(df_new2)

    # df_train = results[results['split'] == 'train']
    
    # df_new_train = df_new['split']== 'train'
    # df_new_test = df_new['split'] == 'test'

    # df_new2 = pd.merge(df_new_train)