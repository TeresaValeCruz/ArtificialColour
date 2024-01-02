import numpy as np
import pandas as pd
from itertools import permutations
import copy
from ColourMaths import sRGB2LUV

phantom_colour = [22, 175, -134]
# phantom_colour = [-10, -100, -150]
minL, maxL = 0.0, 100.0
minU, maxU = -83.06648427886991, 175.01020773938967
minV, maxV = -134.116038927368, 107.40422683605055


def rand_colour(from_rgb=False):
    if from_rgb:
        col_rgb = np.random.randint([0, 0, 0], [255, 255, 255])
        col = sRGB2LUV(col_rgb)
        return col
    col = np.random.randint([minL, minU, minV], [maxL, maxU, maxV])
    return col


indexes = list(permutations([1, 2, 3, 4, 5]))  # all permutations of 5 colours
toggle = list((tuple([int(s) for s in f'{bin(b)[2:]:0>5}']) for b in range(1, 2**5-1)))  # no empty nor complete palette

targets = [tuple(ind) for ind in indexes for b in toggle]
combinations = [tuple(np.multiply(ind, b)) for ind in indexes for b in toggle]  # all incomplete palettes of a set of 5

original_data = pd.read_csv('ghibli_colours.csv')
mask_data = []
input_data = []
target_data = []

for index, row in original_data.iterrows():
    for targ, comb in zip(targets, combinations):
        new_row = [[row[f'L{i-1}'], row[f'U{i-1}'], row[f'V{i-1}']] for i in targ]
        target_data += [copy.deepcopy(new_row)]

        new_inp = [[row[f'L{i-1}'], row[f'U{i-1}'], row[f'V{i-1}']] if bool(i) else rand_colour(True) for i in comb]
        # new_inp = [[row[f'L{i - 1}'], row[f'U{i - 1}'], row[f'V{i - 1}']] if bool(i) else phantom_colour for i in comb]
        input_data += [copy.deepcopy(new_inp)]

        new_msk = [bool(i) for i in comb]
        mask_data += [copy.deepcopy(new_msk)]
    print(index)

np.save('fromRGB_mask.npy', np.array(mask_data, dtype='b'))
np.save('fromRGB_input.npy', np.array(input_data, dtype='i2'))
np.save('fromRGB_target.npy', np.array(target_data, dtype='i2'))
