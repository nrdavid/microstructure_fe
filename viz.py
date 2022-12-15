from typing import List
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

PHASE_MAP = {0:"Matrix", 
             1:"Austinite",
             2:"Martensite/Austenite",
             3:"Precipitate",
             4:"Defect"}

def phase_frequency(df: pd.DataFrame, map: dict=PHASE_MAP)-> None:

    freq = (df[["phase_number", "area"]].
            groupby(by=["phase_number"]).
            count()
            ).to_dict()["area"]
    
    data = sorted(freq.items(), key=lambda x: x[0])
    labels = [x[0] for x in data]
    height = [x[1] for x in data]
    
    plt.bar(labels, height)
    plt.xticks(labels, [map[x] for x in labels], rotation=45)
    plt.ylabel("Frequency")
    plt.title("Phase Frequency in Dataset")
    plt.tight_layout()
    plt.savefig(f"explore/Phase-Frequency-in-Dataset.png")

def scatter(df: pd.DataFrame, col_a: str, col_b: str, phase_map: dict)-> None:

    labels = df["phase_number"].unique()
    _, ax = plt.subplots()
    ax.set_title(f"{col_a} - {col_b}")
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)

    for label in labels:
        phase= PHASE_MAP[label]
        data = df[df["phase_number"]==label][[col_a, col_b]]
        ax.scatter(data[col_a], data[col_b], label=phase, alpha=.35)

    plt.legend(loc=0)
    plt.savefig(f"explore/{col_a} - {col_b} .png")
    plt.close()

def make_plots(df: pd.DataFrame)-> None:

    cache = []
    for i in tqdm(df.columns):
        for c in df.columns:
            if c == "phase_number" or i == "phase_number": 
                continue

            s = set([i,c])
            if any(s.issubset(c) for c in cache):
                continue
            if i==c:
                pass # need to make histograms
            else:
                cache.append(s)
                scatter(df, i, c, PHASE_MAP)

# def plot_predicted_versus_true(data:pd.DataFrame, folder:str) -> None:

#     for group in data.groupby(by='phase'):
#         phase, df = group

#         train = df[df['split']=='train']
#         test = df[df['split']=='test']

#         plt.title(f'Calibration Plot For {phase} Phase')
#         plt.scatter(train['True Area'], train['Predicted Area'], c='blue')
#         plt.scatter(test['True Area'], test['Predicted Area'], c='pink')
#         plt.ylabel('Predicted Total Phase Area')
#         plt.xlabel('True Total Phase Area')
#         plt.ylim(0, )
#         plt.xlim(0, )
#         plt.savefig(f"{folder}/calibration-plot-for-{phase}.png")
#         plt.clf()

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
        plt.savefig(f"{folder}/svc-Residuals-{phase} - dec13 - wo micrograph27.png")
        # plt.show()
        plt.clf()

def plot_confusion_matrices(matrices: List[np.ndarray]) -> None:
    
    for mat, t in zip(matrices,["Training Dataset", "Test Dataset"]):
        print(t, mat)
        sns.heatmap(mat, annot=True).set(title=t)
        plt.show()

if __name__=="__main__":
    # script to create scatter plots.
    # Intentially not using splom functions. These are too crowded.

    FEATURES = ["area", "area_filled", "axis_major_length", "axis_minor_length",
                "eccentricity", "equivalent_diameter_area", "feret_diameter_max",
                "intensity_mean",  "perimeter",
                "solidity"]  # "intensity_min", "intensity_max", ,

    LABEL = ["phase_number"]

    sampleset = pd.read_csv("data/morphological_data.csv", index_col=False).sample(frac=.4)

    cols_to_keep = FEATURES+LABEL

    sampleset.drop(columns=[c for c in sampleset.columns if c not in cols_to_keep], inplace=True)

    if not os.path.exists("explore/"):
        os.mkdir("explore/")

    make_plots(sampleset)

    # phase_frequency(sampleset)