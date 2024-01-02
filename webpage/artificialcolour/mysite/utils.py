import random
import numpy as np
from tensorflow import constant, dtypes
# import tensorflow
from tensorflow.keras.models import load_model
from ColourMaths import LUV2sRGB, HEX2LUV, sRGB2HEX

ranges = constant([100, 260, 243], dtype=dtypes.float32)
mins = constant([0, 84, 135], dtype=dtypes.float32)

generator = load_model('/home/artificialcolour/mysite/models/generator.h5')
# print('ok!')


def norm2luv(gen_out):
    return (gen_out + 1) / 2 * ranges - mins


def luv2norm(luv_colours):
    return (constant(luv_colours, shape=(1, 5, 3), dtype=dtypes.float32) + mins) / ranges * 2 - 1


def generate_palette(list_colours=None, n_try=0):
    if list_colours is None:
        cols = [f'#{random.randint(0, 2**24):06X}' for _ in range(5)]
        mask = [False, False, False, False, False]
    else:
        n = len(list_colours)
        shuffle_cols = list_colours.copy()
        random.shuffle(shuffle_cols)
        cols = [f'#{random.randint(0, 2**24):06X}' for _ in range(5 - n)] + shuffle_cols
        mask = [False]*(5 - n) + [True]*n
    if n_try >= 48:
        return cols

    luv_cols = np.array([HEX2LUV(col) for col in cols])
    norm_input = luv2norm(luv_cols)
    tf_mask = constant(mask, shape=(1, 5), dtype=dtypes.bool)

    gen_out = generator([tf_mask, norm_input], training=False)
    luv_out = norm2luv(gen_out).numpy()[0]
    rgb_out = [LUV2sRGB(col) for col in luv_out]
    for col in rgb_out:
        if col.min() < 0 or col.max() > 255:
            return generate_palette(list_colours, n_try+1)
    hex_out = [sRGB2HEX(col) for col in rgb_out]
    # print(n_try)
    return hex_out


# pal = generate_palette()
# pal = generate_palette(['#80a0d0', '#aabbaa'])
# print(pal)
