import numpy as np
import tensorflow as tf
from ColourMaths import LUV2sRGB

mins = np.array([0, -90, -140])
maxs = np.array([100, 180, 110])
ranges = maxs-mins
err_lut = np.zeros(shape=ranges+4)

for indx in np.ndindex(err_lut.shape):
    print(np.array(indx)-2+mins)
    rgb = LUV2sRGB(np.array(indx)-2+mins)
    if min(rgb) < 0 or max(rgb) > 255:
        err_lut[indx] = 1

avgpool_err = tf.keras.layers.AveragePooling3D(
    pool_size=(5, 5, 5),
    strides=1,
    padding='valid',
    data_format='channels_first'
)(tf.constant([[err_lut]]))
avgpool_err = tf.reshape(avgpool_err, ranges)

np_err = avgpool_err.numpy()
print(np_err)
np.save('cie_err_lut.npy', np_err)
