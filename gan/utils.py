import os
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import matplotlib.pyplot as plt
from ColourMaths import LUV2sRGB

LAMBDA = 1  # 5
GAMMA = 5

loss_object = keras.losses.BinaryCrossentropy(from_logits=True)

# generator_optimizer = tf.keras.optimizers.Adam(0.0025, epsilon=0.1)
# discriminator_optimizer = tf.keras.optimizers.Adam(0.005, epsilon=0.1)
generator_optimizer = tf.keras.optimizers.Adam(0.0025, epsilon=0.0001, beta_1=0.5, amsgrad=True, use_ema=True)
discriminator_optimizer = tf.keras.optimizers.Adam(0.001, epsilon=0.0001, beta_1=0.5, amsgrad=True, use_ema=True)

ranges = tf.constant([100, 260, 243], dtype=tf.dtypes.float32)
mins = tf.constant([0, 84, 135], dtype=tf.dtypes.float32)
err_lut = tf.constant(np.load('cie_err_lut.npy'))

checkpoint_dir = 'checkpoints_dir'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


# display
def array2plot(array):
    """
    given an array of luv colours of a binary mask returns a np array to be displayed
    :param array: of shape (5, 3) or (5, )
    :return: an array of (2, 10, 3)
    """
    phantom = np.array([[[255, 255, 255], [204, 204, 204]], [[204, 204, 204], [255, 255, 255]]])
    white = np.array([[[255, 255, 255], [255, 255, 255]], [[255, 255, 255], [255, 255, 255]]])
    if array.ndim == 1:
        return np.concatenate([
            white * val for val in array
        ], axis=1)
    else:
        rgb_array = np.array([LUV2sRGB(col) for col in array])
        return np.concatenate([
            np.array([[col, col], [col, col]]) if col.min() >= 0 and col.max() < 256 else phantom for col in rgb_array
        ], axis=1)


def init_block_display(nrows=4, ncols=9):
    fig = plt.figure(dpi=100)  # (layout="constrained")
    subfigs = fig.subfigures(nrows, ncols)
    blocks = []
    for k in range(nrows * ncols):
        axs = subfigs[k // ncols, k % ncols].subplots(4, 1, sharex=True, sharey=True,
                                                      gridspec_kw={'wspace': 0, 'hspace': 0})
        for ax in axs:
            ax.axis('off')
        blocks += [axs]
    return fig, blocks


def batch_display_block(blocks, msk_sample, inp_sample, tar_sample, gen_sample, gen_only=False):
    num_pals = len(blocks)
    if not gen_only:
        luv_inp = (inp_sample * 0.5 + 0.5) * ranges - mins
        luv_tar = (tar_sample * 0.5 + 0.5) * ranges - mins
    luv_gen = (gen_sample * 0.5 + 0.5) * ranges - mins
    for k in range(num_pals):
        if not gen_only:
            blocks[k][0].imshow(array2plot(msk_sample[k]), interpolation='none')
            blocks[k][2].imshow(array2plot(luv_inp[k]), interpolation='none')
            blocks[k][1].imshow(array2plot(luv_tar[k]), interpolation='none')
        blocks[k][3].imshow(array2plot(luv_gen[k]), interpolation='none')
    plt.show(block=False)
    plt.pause(1)


# train test split
def data_split(msk, inp, tar, test_ratio=0.2):
    """
    shuffles and cuts the data
    :param msk: tf.tensor, (none, 5)
    :param inp: tf.tensor, (none, 5, 3)
    :param tar: tf.tensor, (none, 5, 3)
    :param test_ratio: value between 0 and 1
    :return: train and test tf.data.dataset of msk, inp and tar
    """
    len_data = msk.shape[0]
    split_ind = int(len_data * test_ratio)  # define the train test cut
    inds = np.arange(len_data)
    np.random.shuffle(inds)

    test_inds = tf.constant(inds[:split_ind], shape=(split_ind, 1))
    train_inds = tf.constant(inds[split_ind:], shape=(len_data - split_ind, 1))

    test_msk, train_msk = tf.gather_nd(msk, test_inds), tf.gather_nd(msk, train_inds)
    test_inp, train_inp = tf.gather_nd(inp, test_inds), tf.gather_nd(inp, train_inds)
    test_tar, train_tar = tf.gather_nd(tar, test_inds), tf.gather_nd(tar, train_inds)

    test_data = tf.data.Dataset.from_tensor_slices((test_msk, test_inp, test_tar))
    train_data = tf.data.Dataset.from_tensor_slices((train_msk, train_inp, train_tar))

    return train_data, test_data


# loss functions
def mod_l1(msk, inp, gen):
    """
    using the input and mask, calculates the generated deviation from context colours
    :param msk: tf.tensor, (none, 5)
    :param inp: tf.tensor, (none, 5, 3)
    :param gen: tf.tensor, (none, 5, 3)
    :return: modified l1 loss, positive integer
    """
    # only consider context values
    abs_error = tf.abs(tf.cast(gen, dtype=inp.dtype) - inp)
    ragged_error = tf.ragged.boolean_mask(abs_error, msk)
    colour_sum = tf.reduce_sum(ragged_error, axis=-1)
    palette_sum = tf.reduce_sum(colour_sum, axis=-1)
    return tf.reduce_mean(palette_sum)


def col_space_err(gen):
    batch_size = gen.shape[0]
    buff = tf.constant([0.0, 90.0, 140.0]) - mins
    corr_gen = tf.cast((gen + 1)/2 * ranges + buff, dtype=tf.dtypes.int32)
    inds = tf.reshape(corr_gen, (batch_size*5, 3))
    clip_max = np.array(err_lut.shape.as_list())-1
    inds = tf.clip_by_value(inds, 0, clip_max.tolist())
    avg_err = tf.reduce_sum(tf.gather_nd(err_lut, inds))/batch_size
    return avg_err


def generator_loss(mask, gen_input, gen_output, disc_generated_output):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = mod_l1(mask, gen_input, gen_output)
    space_loss = col_space_err(gen_output)
    total_gen_loss = tf.cast(gan_loss, dtype=l1_loss.dtype) + GAMMA * space_loss + LAMBDA * l1_loss
    return total_gen_loss, gan_loss, l1_loss, space_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss/2, real_loss, generated_loss


# generator and discriminator nets
def create_generator():
    i0 = layers.Input(shape=(5,), name='mask')
    i1 = layers.Input(shape=(5, 3,), name='input')

    i01 = layers.Reshape((5, 1,), name='mask_reshape')(i0)
    i_conc = layers.Concatenate(name='inputs_concatenate')([i1, i01])

    # encoder
    e11 = layers.Concatenate(axis=1, name='cyclic_concat_e1')([i_conc, i_conc, i_conc])
    e12 = layers.Cropping1D(4, name='cyclic_crop_e1')(e11)
    e13 = layers.Convolution1D(16, 3, name='conv_e1')(e12)
    e14 = layers.BatchNormalization(name='batch_norm_e1')(e13)
    e15 = layers.Activation('elu', name='act_e1')(e14)

    e21 = layers.Concatenate(axis=1, name='cyclic_concat_e2')([e15, e15, e15])
    e22 = layers.Cropping1D(4, name='cyclic_crop_e2')(e21)
    e23 = layers.Convolution1D(32, 3, name='conv_e2')(e22)
    e24 = layers.BatchNormalization(name='batch_norm_e2')(e23)
    e25 = layers.Activation('elu', name='act_e2')(e24)

    # decoder
    d21 = layers.Concatenate(axis=1, name='cyclic_concat_d2')([e25, e25, e25])
    d22 = layers.Cropping1D(4, name='cyclic_crop_d2')(d21)
    d23 = layers.Convolution1DTranspose(16, 3, name='conv_d2')(d22)
    d24 = layers.Cropping1D(2, name='padd_crop_d2')(d23)
    d25 = layers.BatchNormalization(name='batch_norm_d2')(d24)
    d26 = layers.Activation('elu', name='act_d2')(d25)

    s21 = layers.Concatenate(name='concat_s2')([d26, e15])
    s22 = layers.Convolution1DTranspose(16, 1, name='conv_s2')(s21)
    s23 = layers.BatchNormalization(name='batch_norm_s2')(s22)
    s24 = layers.Activation('elu', name='act_s2')(s23)

    d11 = layers.Concatenate(axis=1, name='cyclic_concat_d1')([s24, s24, s24])
    d12 = layers.Cropping1D(4, name='cyclic_crop_d1')(d11)
    d13 = layers.Convolution1DTranspose(4, 3, name='conv_d1')(d12)
    d14 = layers.Cropping1D(2, name='padd_crop_d1')(d13)
    d15 = layers.BatchNormalization(name='batch_norm_d1')(d14)
    d16 = layers.Activation('elu', name='act_d1')(d15)

    s11 = layers.Concatenate(name='concat_s1')([d16, i_conc])
    s12 = layers.Convolution1DTranspose(4, 1, name='conv_s1')(s11)
    s13 = layers.BatchNormalization(name='batch_norm_s1')(s12)
    s14 = layers.Activation('elu', name='act_s1')(s13)

    o1 = layers.Dense(3, name='dense_output')(s14)
    o2 = layers.Activation('tanh', name='act_output')(o1)

    return Model(inputs=[i0, i1], outputs=o2, name='GENERATOR')


def create_discriminator():
    i0 = layers.Input(shape=(5,), name='mask')
    i1 = layers.Input(shape=(5, 3,), name='gen_input')
    i2 = layers.Input(shape=(5, 3,), name='gen_output')

    i01 = layers.Reshape((5, 1,), name='mask_reshape')(i0)
    i_conc = layers.Concatenate(name='inputs_concatenate')([i2, i1, i01])
    # gauss = layers.GaussianNoise(0.05)(i_conc)

    # encoder
    e11 = layers.Concatenate(axis=1, name='cyclic_concat_e1')([i_conc, i_conc, i_conc])
    # e11 = layers.Concatenate(axis=1, name='cyclic_concat_e1')([gauss, gauss, gauss])
    e12 = layers.Cropping1D(4, name='cyclic_crop_e1')(e11)
    e13 = layers.Convolution1D(16, 3, name='conv_e1')(e12)
    e14 = layers.BatchNormalization(name='batch_norm_e1')(e13)
    e15 = layers.Activation('elu', name='act_e1')(e14)

    e21 = layers.Concatenate(axis=1, name='cyclic_concat_e2')([e15, e15, e15])
    e22 = layers.Cropping1D(4, name='cyclic_crop_e2')(e21)
    e23 = layers.Convolution1D(32, 3, name='conv_e2')(e22)
    e24 = layers.BatchNormalization(name='batch_norm_e2')(e23)
    e25 = layers.Activation('elu', name='act_e2')(e24)

    # decoder
    d21 = layers.Concatenate(axis=1, name='cyclic_concat_d2')([e25, e25, e25])
    d22 = layers.Cropping1D(4, name='cyclic_crop_d2')(d21)
    d23 = layers.Convolution1DTranspose(16, 3, name='conv_d2')(d22)
    d24 = layers.Cropping1D(2, name='padd_crop_d2')(d23)
    d25 = layers.BatchNormalization(name='batch_norm_d2')(d24)
    d26 = layers.Activation('elu', name='act_d2')(d25)

    s21 = layers.Concatenate(name='concat_s2')([d26, e15])
    s22 = layers.Convolution1DTranspose(16, 1, name='conv_s2')(s21)
    s23 = layers.BatchNormalization(name='batch_norm_s2')(s22)
    s24 = layers.Activation('elu', name='act_s2')(s23)

    d11 = layers.Concatenate(axis=1, name='cyclic_concat_d1')([s24, s24, s24])
    d12 = layers.Cropping1D(4, name='cyclic_crop_d1')(d11)
    d13 = layers.Convolution1DTranspose(5, 3, name='conv_d1')(d12)
    d14 = layers.Cropping1D(2, name='padd_crop_d1')(d13)
    d15 = layers.BatchNormalization(name='batch_norm_d1')(d14)
    d16 = layers.Activation('elu', name='act_d1')(d15)

    s11 = layers.Concatenate(name='concat_s1')([d16, i_conc])
    # s11 = layers.Concatenate(name='concat_s1')([d16, gauss])
    s12 = layers.Convolution1DTranspose(4, 1, name='conv_s1')(s11)
    s13 = layers.BatchNormalization(name='batch_norm_s1')(s12)
    s14 = layers.Activation('elu', name='act_s1')(s13)

    o1 = layers.Dense(1, name='dense_output')(s14)

    return Model(inputs=[i0, i1, i2], outputs=o1, name='DISCRIMINATOR')


# train test and fit loops
@tf.function
def test_step(batch, generator, discriminator):
    msk_batch = batch[0]
    inp_batch = batch[1]
    tar_batch = batch[2]

    gen_out = generator([msk_batch, inp_batch], training=False)

    disc_real_out = discriminator([msk_batch, inp_batch, tar_batch], training=False)
    disc_gen_out = discriminator([msk_batch, inp_batch, gen_out], training=False)

    gen_total_loss, gan_loss, l1_loss, cie_loss = generator_loss(msk_batch, inp_batch, gen_out, disc_gen_out)
    disc_total_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_out, disc_gen_out)

    return gen_total_loss, gan_loss, l1_loss, cie_loss, disc_total_loss, disc_real_loss, disc_gen_loss


@tf.function
def train_step(batch, generator, discriminator, train_gen=True, train_disc=True):
    msk_batch = batch[0]
    inp_batch = batch[1]
    tar_batch = batch[2]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_out = generator([msk_batch, inp_batch], training=train_gen)

        disc_real_out = discriminator([msk_batch, inp_batch, tar_batch], training=train_disc)
        disc_gen_out = discriminator([msk_batch, inp_batch, gen_out], training=train_disc)

        gen_total_loss, gan_loss, l1_loss, cie_loss = generator_loss(msk_batch, inp_batch, gen_out, disc_gen_out)
        disc_total_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_out, disc_gen_out)

    if train_gen:
        gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    if train_disc:
        disc_gradients = disc_tape.gradient(disc_total_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_total_loss, gan_loss, l1_loss, cie_loss, disc_total_loss, disc_real_loss, disc_gen_loss


def fit(generator, discriminator, msk, inp, tar, ckpt, test_ratio=0.2, batch_size=10000, epochs=1, display=True):
    # train_data, test_data = data_split(msk, inp, tar, test_ratio=test_ratio)
    df = pd.DataFrame(columns=['dt', 'epoch', 'step', 'gen_total', 'gen_adv', 'gen_context', 'gen_cie', 'disc_total', 'disc_real', 'disc_synthetic', 'test_gen', 'test_disc'])
    dt = datetime.now()
    df.to_csv(f'{str(dt)}.csv', index=False)
    print(datetime.now())

    if display:
        num_rows = 6
        num_cols = 5
        num_pals = num_rows * num_cols
        fig, block_axs = init_block_display(num_rows, num_cols)
        display_inds = np.random.randint(0, msk.shape[0], num_pals)
        msk_sample = tf.gather(msk, indices=display_inds).numpy()
        inp_sample = tf.gather(inp, indices=display_inds).numpy()
        tar_sample = tf.gather(tar, indices=display_inds).numpy()
        gen_sample = generator([msk_sample, inp_sample], training=False).numpy()
        batch_display_block(block_axs, msk_sample, inp_sample, tar_sample, gen_sample)
        plt.savefig('train_images2/first.png', dpi=300)

    head1 = '| epoch |  step  |     progress bar     ||                  generator                  ||         discriminator          ||          test         |\n'
    head2 = '|       |        |                      ||   total   | adversarial | context |   cie   ||  total  |  real  |  generated  ||    gen    |    disc   |\n'
    print(head1 + head2, end='', flush=True)

    for ep in tf.range(epochs):
        train_data, test_data = data_split(msk, inp, tar, test_ratio=test_ratio)
        k = 0
        train_batches = train_data.batch(batch_size=batch_size)
        test_batches = iter(test_data.batch(batch_size=batch_size))
        num_bats = train_batches.cardinality()

        for batch in train_batches:
            # if k % 32 == 0 or gl3<1:
            #     gen_loss, gl1, gl2, gl3, disc_loss, dl1, dl2 = train_step(batch, generator, discriminator)
            # else:
            #     gen_loss, gl1, gl2, gl3, disc_loss, dl1, dl2 = train_step(batch, generator, discriminator, train_disc=False)
            gen_loss, gl1, gl2, gl3, disc_loss, dl1, dl2 = train_step(batch, generator, discriminator)

            if k % 4 == 0:  # train/test 0.8/0.2 = 4
              test_batch = next(test_batches)
              test_gl, _, _, _, test_dl, _, _ = test_step(test_batch, generator, discriminator)

            k += 1
            bar = (k * 20 // num_bats) * '\u2588' + (20 - k * 20 // num_bats) * '\u2591'
            print(
                f'\r| {ep: 5} | {k: 6} | {bar} || {gen_loss: 9.3f} | {gl1: 11.3f} | {gl2: 7.3f} | {gl3: 7.3f} || {disc_loss: 7.3f} | {dl1: 6.3f} | {dl2: 11.3f} || {test_gl: 9.3f} | {test_dl: 9.3f} |',
                end='', flush=True
            )

            df_row = {
                'dt': [str(datetime.now())],
                'epoch': [f'{ep}'],
                'step': [f'{k}'],
                'gen_total': [f'{gen_loss}'],
                'gen_adv': [f'{gl1}'],
                'gen_context': [f'{gl2}'],
                'gen_cie': [f'{gl3}'],
                'disc_total': [f'{disc_loss}'],
                'disc_real': [f'{dl1}'],
                'disc_synthetic': [f'{dl2}'],
                'test_gen': [f'{test_gl}'],
                'test_disc': [f'{test_dl}']
            }
            df = pd.DataFrame(df_row)
            df.to_csv(f'{str(dt)}.csv', index=False, mode='a', header=False)

            if display and k % 32 == 0:
            # if display and k % 128 == 0:
                gen_sample = generator([msk_sample, inp_sample], training=False).numpy()
                batch_display_block(block_axs, msk_sample, inp_sample, tar_sample, gen_sample, gen_only=True)
                plt.savefig(f'train_images2/ep{ep:03}step{k:05}.png', dpi=300)

        ckpt.save(file_prefix=checkpoint_prefix)
        print('')


gen = create_generator()
disc = create_discriminator()

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen,
                                 discriminator=disc)
if __name__ == '__main__':
    MSK = tf.constant(np.load('fromRGB_mask.npy'), dtype=tf.dtypes.bool)
    INP = (tf.constant(np.load('fromRGB_input.npy'), dtype=tf.dtypes.float32) + mins) / ranges * 2 - 1
    TAR = (tf.constant(np.load('fromRGB_target.npy'), dtype=tf.dtypes.float32) + mins) / ranges * 2 - 1

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)

    # checkpoint.restore(manager.latest_checkpoint)
    # if manager.latest_checkpoint:
    #     print('restored')

    # fit(gen, disc, MSK, INP, TAR, epochs=30, batch_size=11000, ckpt=checkpoint)
    print(col_space_err(TAR))