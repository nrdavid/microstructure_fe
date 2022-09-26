import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops


if __name__=="__main__":
    
    PHASE_MAP = {0:"Matrix", 
                 1:"Austinite",
                 2:"Martensite/Austenite",
                 3:"Precipitate",
                 4:"Defect"}

    morphological_data = []
    for pic in tqdm(os.listdir(r"MetalDAM/cropped_grayscale")):

        im_og = imread(f"MetalDAM/cropped_grayscale/{pic}", as_gray=True)
        im_lb = imread(f"MetalDAM/labels/{pic.replace('jpg','png')}", as_gray=True)
        
        for k in np.unique(im_lb):
            mask = np.select([im_lb==k, im_lb!=k], [1, 0], im_og)
            labels = label(mask)
            
            # Uncomment to see phase comparision.
            # fig, ax = plt.subplots(1,2)
            # ax[0].set_title(f"labeled {PHASE_MAP[k]}")
            # ax[0].imshow(mask)
            # ax[1].set_title(f"Raw {PHASE_MAP[k]}")
            # ax[1].imshow(labels)
            # # plt.title(f"{pic}\n count: {count} , phase {k}")
            # plt.show()

            props = ["area", "area_filled", "axis_major_length", "axis_minor_length",
                    "eccentricity", "equivalent_diameter_area", "feret_diameter_max",
                    "intensity_max", "intensity_mean", "intensity_min", "perimeter",
                    "solidity"]

            properties = regionprops(labels, im_og)

            results = pd.DataFrame.from_dict(
            {l: {p: properties[l][p] for p in props} for l in range(1, labels.max())}, orient='index'
            )
            try:
                results["phase_number"] =  k
                results["phase"] = PHASE_MAP[k]
                results["picture"] = pic
                results["pic_height"] = im_og.shape[0]
                results["pic_width"] = im_og.shape[1]
            except:
                del results
                continue

            morphological_data.append(results)

pd.concat(morphological_data).to_csv("morphological_data.csv")