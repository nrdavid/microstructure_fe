import os
import numpy as np
import pandas as pd
import sqlite3
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops

PROPS = [
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


def get_pixel_to_area_conversion(im: str, db:str=r"MetalDAM/MetalDAM_metadata.sql") -> float:

    with sqlite3.connect(db) as conn:
        cur = conn.cursor()

        sql = f"""SELECT pixels_per_micron FROM micrograph WHERE image_path LIKE '%{im}';"""
        cur.execute(sql)
        conversion = cur.fetchone()
        conversion = float(conversion[0])
    return conversion

if __name__=="__main__":
    # Once DBSCAN is implemented, create function to get feature properties.
    
    PHASE_MAP = {0:"Matrix", 
                 1:"Austinite",
                 2:"Martensite/Austenite",
                 3:"Precipitate",
                 4:"Defect"}

    morphological_data = []
    for pic in tqdm(os.listdir(r"MetalDAM/cropped_grayscale")):

        im_og = imread(f"MetalDAM/cropped_grayscale/{pic}", as_gray=True)
        im_lb = imread(f"MetalDAM/labels/{pic.replace('jpg','png')}", as_gray=True)

        conversion = get_pixel_to_area_conversion(pic)
        
        for k in np.unique(im_lb):
            #  mask = np.select([im_lb==k, im_lb!=k], [True, False], im_og)
            mask = im_lb == k
            labels = label(mask, connectivity=2)

            # Uncomment to see phase comparision.
            # fig, ax = plt.subplots(1,2)
            # ax[0].set_title(f"labeled {PHASE_MAP[k]}")
            # ax[0].imshow(mask)
            # ax[1].set_title(f"Raw {PHASE_MAP[k]}")
            # ax[1].imshow(labels)
            # # plt.title(f"{pic}\n count: {count} , phase {k}")
            # plt.show()

            # props = ["area", "area_filled", "axis_major_length", "axis_minor_length",
            #         "eccentricity", "equivalent_diameter_area", "feret_diameter_max",
            #         "intensity_max", "intensity_mean", "intensity_min", "perimeter",
            #         "solidity"]

            properties = regionprops(labels, im_og)

            results = pd.DataFrame.from_dict(
            {l: {p: properties[l][p] for p in PROPS} for l in range(1, labels.max())}, orient='index'
            )
            try:
                # change to test if results.empty then del results.
                results["phase_number"] =  k
                results["phase"] = PHASE_MAP[k]
                results["picture"] = pic.split('.')[0]
                results["pic_height"] = im_og.shape[0]
                results["pic_width"] = im_og.shape[1]
            except:
                del results
                continue

            area_cols = [c for c in results.columns if re.search(r"area|diameter|length",c)]
            for col in area_cols:
                results[col] = results[col] / conversion

            morphological_data.append(results)

if not os.path.exists("data"):
        os.mkdir("data/")

pd.concat(morphological_data).to_csv("data/morphological_data.csv", index=False)