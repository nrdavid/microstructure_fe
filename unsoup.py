from skimage.io import imread
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from skimage.measure import regionprops_table
from skimage.measure import _regionprops
from skimage.segmentation import mark_boundaries
from kneed import KneeLocator
import cv2
import skimage
import matplotlib
import warnings
from collections import defaultdict
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from os.path import exists

PROPS = _regionprops.PROPS
PROP_DTYPES = _regionprops.COL_DTYPES
numerical_props = []
for x in PROPS.values():
    if PROP_DTYPES[x] in (int, float):
        numerical_props.append(x)

def felzenszwalb_img(img):
    img = img_as_float(img)
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=500)
    return segments_fz

def reduce_img(img, segments):
    reduced = np.zeros(np.max(segments))
    for i in tqdm(range(np.max(segments))):
        island = np.where(segments == i)
        summer = 0
        num_pixels = 0
        for x,y in zip(island[0], island[1]):
            summer += img[x, y]
            num_pixels += 1
        reduced[i] = summer
    return reduced

def map_reduced_labels_to_og(labels, segments, df):
    seg_to_cluster_label = dict()
    for df_label, cluster_label in zip(df['label'], labels):
        seg_to_cluster_label[df_label-1] = cluster_label
    seg_copy = np.copy(segments)
    for i in range(len(labels)):
        l = seg_to_cluster_label[i]
        seg_copy[segments == i] = l
    return seg_copy

def get_morphological_data(img, segments):
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
    props = regionprops_table(segments, intensity_image=img, properties=PROPS)
    # print("segs")
    # print(segments)
    # print("img")
    # print(img)
    df = pd.DataFrame.from_dict(props)
    return df

def do_kmeans(im_og, pic_name):
    im_og_X = np.zeros((im_og.shape[0]*im_og.shape[1], 1))
    count = 0
    for (i, j), v in np.ndenumerate(im_og):
        im_og_X[count, :] = im_og[i, j]
        count += 1
    distorsions = []
    kmeans_list = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(im_og_X)
        kmeans_list.append(kmeans)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    distorsions = distorsions/np.linalg.norm(distorsions)
    kneedle = KneeLocator(distorsions, range(2, 8), S=1.0, curve='convex', direction='decreasing')
    plt.plot(kneedle.x_normalized, kneedle.y_normalized, label="Distortions")
    plt.plot(kneedle.x_difference, kneedle.y_difference, label='Difference')
    # plt.plot(kneedle.x, kneedle.y, label="Distortions")
    # plt.plot(kneedle.x, kneedle.y, label='Difference')
    plt.vlines(kneedle.norm_elbow, 0, 1, linestyles='--', label=f"S = 1.0, K = {round(kneedle.elbow_y, 2)}")
    plt.xlabel('Distortions')
    plt.ylabel('K')
    plt.legend()
    plt.grid(True)
    plt.title('Elbow curve')
    plt.savefig(f'nick_all_pics/{pic_name}_elbow_curve.png')
    im_show_labels = np.copy(im_og)
    count =  0
    for (i, j), v in np.ndenumerate(im_og):
        im_show_labels[i, j] = kmeans_list[kneedle.elbow_y-2].labels_[count]
        count += 1
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(im_show_labels)
    plt.title(f"Kmeans Generated Labels with K = {kneedle.elbow_y}.")
    plt.colorbar()
    plt.savefig(f'nick_all_pics/{pic_name}_kmeans_labels.png')
    plt.close()
    return im_show_labels


def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='gray')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='gray')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()
    plt.close()

def do_dbscan(img, pic_name):
    img_X = np.zeros((img.shape[0]*img.shape[1], 1))
    count = 0
    for (i, j), v in np.ndenumerate(img):
        img_X[count, :] = img[i, j]
        count += 1
    db = DBSCAN(min_samples=100, eps=1)
    db.fit(img_X)
    im_show_labels = np.copy(img)
    count =  0
    for (i, j), v in np.ndenumerate(img):
        im_show_labels[i, j] = db.labels_[count]
        count += 1
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(im_show_labels)
    plt.title(f"DBSCAN Generated Labels")
    plt.colorbar()
    plt.savefig(f'{pic_name}_DBSCAN_labels.png')
    plt.close()

def plot_quad(img_og, img_new, img_label, hand_labeled, fname, filter_method='None', cluster_method = 'None'):
    #fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout()
    # top left
    axs[0, 0].imshow(img_og, cmap='Greys')
    axs[0, 0].set_title("Original Microstructure")
    # top right
    axs[0, 1].imshow(img_new, cmap='Greys')
    axs[0, 1].set_title(f"Augmented Image by {filter_method}")
    # bottom left
    axs[1, 0].imshow(img_label, cmap='viridis')
    axs[1, 0].set_title(f"Image Labels After {cluster_method} Clustering")
    # bottom right
    axs[1, 1].imshow(hand_labeled, cmap='viridis')
    axs[1, 1].set_title(f"Hand Labeled Image")
    plt.savefig(fname)
    plt.close()

def gen_fez_segment(pic1, kmeans, fname):
    img = imread(f"MetalDAM/cropped_grayscale/{pic1}", as_gray=True)
    segmented_fz_og = felzenszwalb_img(img)
    segmented_fz_kmeans = felzenszwalb_img(kmeans)
    fz_segs = mark_boundaries(img, segmented_fz_og)
    fz_segs_kmeans = mark_boundaries(kmeans, segmented_fz_kmeans)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.tight_layout()
    # top left
    axs[0, 0].imshow(img, cmap='viridis')
    axs[0, 0].set_title("Original Microstructure")
    # top right
    axs[0, 1].imshow(fz_segs, cmap='viridis')
    axs[0, 1].set_title(f"Segnmented by Felzenszwalb Method")
    # bottom left
    axs[1, 0].imshow(kmeans, cmap='viridis')
    axs[1, 0].set_title(f"Image Labels After KMeans Clustering")
    # bottom right
    axs[1, 1].imshow(fz_segs_kmeans, cmap='viridis')
    axs[1, 1].set_title(f"Segnmented by Felzenszwalb Method after KMeans")
    plt.savefig(fname)
    plt.close()
    return segmented_fz_og


def gen_gaussian_blur_quad(pic1, pic2, fname):
    img = imread(f"MetalDAM/cropped_grayscale/{pic1}", as_gray=True)
    img_new = skimage.filters.gaussian(img)
    img_hand_labeled = imread(f"MetalDAM/labels/{pic2}")
    img_labeled = do_kmeans(img_new, pic1)
    #fname = "gaussian_blur_quad.png"
    plot_quad(img, img_new, img_labeled, img_hand_labeled, fname, filter_method='Gaussian Blur', cluster_method="KMeans")
    return img_labeled

def assign_islands_labels(fz_seg, img_labeled):
    d_ret = dict()
    for i in range(np.min(fz_seg), np.max(fz_seg)+1):
        island = np.where(fz_seg == i)
        d = defaultdict(int)
        for x,y in zip(island[0], island[1]):
            d[int(img_labeled[x, y])] += 1
        max_lab = max(d, key=d.get)
        d_ret[i] = max_lab
    return d_ret

def gen_props(fz_seg, img, img_labeled):
    fz_seg += 1
    df_props = get_morphological_data(img, fz_seg)
    df_props_og = df_props.copy(deep=True)
    labs = []
    ail = assign_islands_labels(fz_seg, img_labeled)
    for i, row in enumerate(df_props.iterrows()):
        j = i+1
        labs.append(ail[j])
    df_props['label_majority'] = labs
    df_props = df_props.dropna(axis=1)
    return df_props, df_props_og

def do_tsne(df_props, fname):
    np_props = df_props.to_numpy()
    tsne = TSNE(n_components=2, verbose=1, init='random', perplexity=20)
    dat = np_props[:, 0:-1]
    scaler = StandardScaler()
    #scaler = PowerTransformer()
    scaled_dat = scaler.fit_transform(dat)
    z = tsne.fit_transform(scaled_dat)
    df = pd.DataFrame()
    df["y"] = np_props[:, -1].astype(int)
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    plt.clf()
    sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", max(df.y.tolist())+1),
                data=df)
    sns_plot.set(title="Morphology Data Projection")
    sns_plot.figure.savefig(fname)

def do_pca_kmeans(df_props):
    np_props = df_props.to_numpy()
    pca = PCA(2)
    pca_x = pca.fit_transform(np_props[:, 0:-1])
    #kmeans = KMeans(n_clusters=np.max(np_props[:, -1])+1)
    plt.clf()
    plt.scatter(pca_x[:, 0], pca_x[:, 1], c=np_props[:, -1])
    plt.legend()
    plt.savefig("pca_plot.png")
    

def main():
    df_tsne = pd.DataFrame()
    df_props_og_all = pd.DataFrame()
    for pic1, pic2 in zip(sorted(os.listdir(r"MetalDAM/cropped_grayscale")), sorted(os.listdir(r"MetalDAM/labels"))):
        img = imread(f"MetalDAM/cropped_grayscale/{pic1}", as_gray=True)
        img_labeled = gen_gaussian_blur_quad(pic1, pic2, f"nick_all_pics/gaussian_blur_quad_{pic1}")
        fz_seg = gen_fez_segment(pic1, img_labeled, f"nick_all_pics/felzenszwalb_{pic1}")
        df_props, df_props_og = gen_props(fz_seg, img, img_labeled)
        df_props_og['picture'] = pic1
        df_props_og_all = pd.concat([df_props_og_all, df_props_og])
    df_props_og_all.to_pickle("tom_final.pkl")
    pass

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
    main()