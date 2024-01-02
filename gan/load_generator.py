import random
import numpy as np
from tensorflow import constant, dtypes
from tensorflow.keras.models import load_model
from tensorflow import train
from utils import checkpoint_dir, checkpoint, mins, ranges
from utils import gen as generator
from ColourMaths import LUV2sRGB, sRGB2LUV, HEX2LUV, sRGB2HEX

manager = train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
checkpoint.restore(manager.latest_checkpoint)

generator.save("generator.keras")
generator.save("generator.h5")

# generator = load_model('generator.h5')

#
# def norm2luv(gen_out):
#     return (gen_out + 1) / 2 * ranges - mins
#
#
# def luv2norm(luv_colours):
#     return (constant(luv_colours, shape=(1, 5, 3), dtype=dtypes.float32) + mins) / ranges * 2 - 1
#
#
# def generate_palette(list_colours=None, n_try=0):
#     if list_colours is None:
#         cols = [f'#{random.randint(0, 2**24):06X}' for _ in range(5)]
#         mask = [False, False, False, False, False]
#     else:
#         n = len(list_colours)
#         cols = [f'#{random.randint(0, 2**24):06X}' for _ in range(5 - n)] + list_colours
#         mask = [False]*(5 - n) + [True]*n
#
#     luv_cols = np.array([HEX2LUV(col) for col in cols])
#     norm_input = luv2norm(luv_cols)
#     tf_mask = constant(mask, shape=(1, 5), dtype=dtypes.bool)
#
#     gen_out = generator([tf_mask, norm_input], training=False)
#     # print(norm_input)
#     # print(gen_out)
#     luv_out = norm2luv(gen_out).numpy()[0]
#     rgb_out = [LUV2sRGB(col) for col in luv_out]
#     # print(rgb_out)
#     for col in rgb_out:
#         if col.min() < 0 or col.max() > 255:
#             return generate_palette(list_colours, n_try+1)
#     hex_out = [sRGB2HEX(col) for col in rgb_out]
#     print(n_try)
#     return hex_out
#
#
# pal = generate_palette()
# # pal = generate_palette(['#80a0d0', '#aabbaa'])
# print(pal)
