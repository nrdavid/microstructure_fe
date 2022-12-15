import os
import numpy as np
import pandas as pd
import sqlite3
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

PHASE_MAP = {
    0:"Matrix", 
    1:"Austinite",
    2:"Martensite/Austenite",
    3:"Precipitate",
    4:"Defect"
}

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

def get_true_phase_amount(im: str, p:int, db:str=r"MetalDAM/MetalDAM_metadata.sql") -> float:

    with sqlite3.connect(db) as conn:
        cur = conn.cursor()

        col = f"label{int(p)}"

        sql = f"""SELECT {col} FROM micrograph WHERE image_path LIKE '%{im}';"""
        cur.execute(sql)
        phase_amount = cur.fetchone()
    return phase_amount[0]

def get_pixel_to_area_conversion(im: str, db:str=r"MetalDAM/MetalDAM_metadata.sql") -> float:

    with sqlite3.connect(db) as conn:
        cur = conn.cursor()

        sql = f"""SELECT pixels_per_micron FROM micrograph WHERE image_path LIKE '%{im}';"""
        cur.execute(sql)
        conversion = cur.fetchone()
        conversion = float(conversion[0])
    return conversion

if __name__=="__main__":

    morphological_data = [] # list of dataframes

    unit_test = {}
    c = 0
    for pic in tqdm(os.listdir(r"MetalDAM/cropped_grayscale")): # iterate through all pictures

        im_og = imread(f"MetalDAM/cropped_grayscale/{pic}", as_gray=True)
        im_lb = imread(f"MetalDAM/labels/{pic.replace('jpg','png')}", as_gray=True)
        
        conversion = get_pixel_to_area_conversion(pic) # query sqlite database

        if pic == "micrograph27.jpg":
            continue

        for k in np.unique(im_lb): # iterate through unique phase labels in the image.
            mask = im_lb == k
            mask[0, :] = False
            mask[:,0] = False
            mask[-1,:] = False
            mask[:, -1] = False

            p_amount = get_true_phase_amount(pic, k)

            labels = label(mask, connectivity=2) # regions as found via connectivity of nearest neighbor 2.
            
            properties = regionprops(labels, im_og)
            
            results = pd.DataFrame.from_dict(
            {l: {p: properties[l][p] for p in PROPS} for l in range(0, labels.max())}, orient='index'
            )
            
            try:
                phase_in_image1 = results.area.sum()
            except:
                phase_in_image1 = 0

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

            area_cols = [c for c in results.columns if re.search(r"area|diameter|length|perimeter",c)]
            for col in area_cols:
                if col.__contains__('area') and (col != "equivalent_diameter_area"):
                    results[col] = results[col] / conversion**2
                else:
                    results[col] = results[col] / conversion

            
            try:
                phase_in_image2 = results.area.sum()
            except:
                phase_in_image2 = 0

            unit_test[c] = {
                'pic': pic,
                'pixel_to_micron': conversion,
                'phase': PHASE_MAP[k],
                'true px': p_amount,
                'true um': round((p_amount / conversion**2), 2),
                'masked px': np.count_nonzero(mask),
                'masked um': np.round((np.count_nonzero(mask)/conversion**2), 2),
                'post_label px': phase_in_image1,
                'post_label um': round((phase_in_image1 / conversion**2), 2),
                'post_conversion um': round(phase_in_image2, 2)
            }
            c += 1              
            
            morphological_data.append(results)

if not os.path.exists("data"):
        os.mkdir("data/")

pd.concat(morphological_data).to_csv("data/morphological_data.csv", index=False)

# uncomment to test for unit conversion
# df_unit_test = pd.DataFrame.from_dict(unit_test, orient='index').round(3)
# df_unit_test['<= True Area'] = df_unit_test['post_conversion um'] <= df_unit_test['true um']
# df_unit_test['< True Area'] = df_unit_test['post_conversion um'] < df_unit_test['true um']
# df_unit_test['difference'] = abs(df_unit_test['true um'] - df_unit_test['post_conversion um'])
# df_unit_test['pct diff'] = (df_unit_test['post_conversion um'] - df_unit_test['true um']) / df_unit_test['true um']
# print(df_unit_test)
# df_unit_test.to_csv('data/units_troubleshoot_3.csv')
