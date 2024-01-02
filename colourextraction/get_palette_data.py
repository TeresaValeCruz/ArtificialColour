import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from ColourMaths import HEX2sRGB, sRGB2LUV
from PaletteExtractor import colour_palette_ratio
import time

n_colours = 5

cols = ['n_image']
for k in range(n_colours):
    cols += [f'hex{k}', f'L{k}', f'U{k}', f'V{k}', f'size{k}']
dt = pd.DataFrame(columns=cols)


def get_palette(image_path):
    # this function takes an image number and return a dict with corresponding colour palette

    # image_path = f'{image_folder}/{image_name}' #  'studio-ghibli-images/' + image_name + '.jpg'
    print('extracting palette from ' + image_path)

    palette = colour_palette_ratio(image_path, n_clusters=n_colours)

    luv_palette = np.array([sRGB2LUV(HEX2sRGB(c)) for c in palette.keys()])
    # new_row = dict()
    new_row = {'n_image': image_path[-8:-4]}

    for col in range(n_colours):
        new_row[f'hex{col}'] = list(palette.keys())[col]
        new_row[f'size{col}'] = list(palette.values())[col]
        new_row[f'L{col}'] = luv_palette[col, 0]
        new_row[f'U{col}'] = luv_palette[col, 1]
        new_row[f'V{col}'] = luv_palette[col, 2]

    print(new_row)
    return new_row


files = os.listdir('studio_ghibli')
files.sort()
paths = [f'studio_ghibli/{file}' for file in files]

start = time.time()

p = Pool()
results = p.map(get_palette, paths)

p.close()
p.join()

end = time.time()

run_time = end - start
print(f'process took {int(run_time // 3600)} hours and {int(run_time // 60) % 60} minutes')

dt = pd.concat([dt, pd.DataFrame(results)], axis=0, ignore_index=True)
dt.to_csv('ghibli_colours.csv', index=False)
