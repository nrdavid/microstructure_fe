import os
import numpy as np
import sqlite3
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from tqdm import tqdm

if __name__=="__main__":
    # Thomas Korejszaa
    # Script to crop off SEM metadata from image.
    
    FOLDER = "MetalDAM/images"
    if not os.path.isdir(r"MetalDAM/cropped_grayscale/"):
        os.mkdir("MetalDAM/cropped_grayscale/")

    db = r"MetalDAM/MetalDAM_metadata.sql"
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        print("Cropping Images..")
        for pic in tqdm(os.listdir(FOLDER)):
            
            cur.execute(f"SELECT height, width FROM micrograph WHERE image_path LIKE 'images/{pic}'")
            height, width = cur.fetchone() 

            im_cropped = imread(os.path.join(FOLDER, pic), as_gray=True)[:height, :width]

            imsave(f"MetalDAM/cropped_grayscale/{pic}", img_as_ubyte(im_cropped))