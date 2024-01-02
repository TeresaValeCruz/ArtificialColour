import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from ColourMaths import sRGB2LUV, LUV2sRGB, sRGB2HEX, HEX2sRGB


def colour_palette_ratio(image_path, n_clusters=5, to_luv=True, rs=0):
    img = Image.open(image_path)
    img = np.array(img)
    size = img.shape
    pixels = img.reshape(size[0] * size[1], size[2])
    if to_luv:
        pixels = np.apply_along_axis(sRGB2LUV, 1, pixels)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init='auto', random_state=rs, batch_size=256)
    kmeans.fit(pixels)
    centres = kmeans.cluster_centers_
    if to_luv:
        centres = np.apply_along_axis(LUV2sRGB, 1, centres)
    labels = kmeans.labels_
    label_count = [list(labels).count(k) / len(labels) for k in range(n_clusters)]
    return {sRGB2HEX(centres[k]): label_count[k] for k in range(n_clusters)}


def colour_palette(image_path, n_clusters=5, to_luv=True):
    hex_palette = list(colour_palette_ratio(image_path, n_clusters, to_luv))
    return np.array([HEX2sRGB(c) for c in hex_palette])
