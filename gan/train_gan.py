import numpy as np
import tensorflow as tf
from utils import checkpoint_dir, checkpoint, gen, disc, fit, mins, ranges

MSK = tf.constant(np.load('fromRGB_mask.npy'), dtype=tf.dtypes.bool)
INP = (tf.constant(np.load('fromRGB_input.npy'), dtype=tf.dtypes.float32) + mins) / ranges * 2 - 1
TAR = (tf.constant(np.load('fromRGB_target.npy'), dtype=tf.dtypes.float32) + mins) / ranges * 2 - 1

manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print('restored!')

fit(gen, disc, MSK, INP, TAR, epochs=20, batch_size=11000, ckpt=checkpoint)
