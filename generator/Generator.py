import scipy
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from scipy import misc
from keras.layers import BatchNormalization, Convolution2D, Dense, LeakyReLU, \
    Input, MaxPooling2D, merge, Reshape, UpSampling2D
from keras.models import Model
import tensorflow as tf

NUM_YALE_POSES = 10


# ---- Enum classes for vector descriptions

class Emotion:
    angry = [1., 0., 0., 0., 0., 0., 0., 0.]
    contemptuous = [0., 1., 0., 0., 0., 0., 0., 0.]
    disgusted = [0., 0., 1., 0., 0., 0., 0., 0.]
    fearful = [0., 0., 0., 1., 0., 0., 0., 0.]
    happy = [0., 0., 0., 0., 1., 0., 0., 0.]
    neutral = [0., 0., 0., 0., 0., 1., 0., 0.]
    sad = [0., 0., 0., 0., 0., 0., 1., 0.]
    surprised = [0., 0., 0., 0., 0., 0., 0., 1.]

    mixed = [1.0, 0., 0., 0., 1.0, 0., 0., 1.0]

    @classmethod
    def length(cls):
        return len(Emotion.neutral)


def log10(x):
    """
    there is not direct implementation of log10 in TF.
    But we can create it with the power of calculus.
    Args:
        x (array): input array

    Returns: log10 of x

    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * log10(K.mean(K.square(y_pred - y_true)))


def build_model(identity_len=57, orientation_len=2, lighting_len=4,
                emotion_len=8, pose_len=NUM_YALE_POSES,
                initial_shape=(5, 4), deconv_layers=5, num_kernels=None,
                optimizer='adam', use_yale=False):
    """
    Builds a deconvolution FaceGen model.

    Args (optional):
        identity_len (int): Length of the identity input vector.
        orientation_len (int): Length of the orientation input vector.
        emotion_len (int): Length of the emotion input vector.
        initial_shape (tuple<int>): The starting shape of the deconv. network.
        deconv_layers (int): How many deconv. layers to use. More layers
            gives better resolution, although requires more GPU memory.
        num_kernels (list<int>): Number of convolution kernels for each layer.
        optimizer (str): The optimizer to use. Will only use default values.
    Returns:
        keras.Model, the constructed model.
    """

    if num_kernels is None:
        num_kernels = [128, 128, 96, 96, 32, 32, 16]

    # TODO: Parameter validation

    identity_input = Input(shape=(identity_len,), name='identity')

    if use_yale:
        lighting_input = Input(shape=(lighting_len,), name='lighting')
        pose_input = Input(shape=(pose_len,), name='pose')
    else:
        orientation_input = Input(shape=(orientation_len,), name='orientation')
        emotion_input = Input(shape=(emotion_len,), name='emotion')

    # Hidden representation for input parameters

    fc1 = LeakyReLU()(Dense(512)(identity_input))
    fc2 = LeakyReLU()(Dense(512)(lighting_input if use_yale else orientation_input))
    fc3 = LeakyReLU()(Dense(512)(pose_input if use_yale else emotion_input))

    params = merge([fc1, fc2, fc3], mode='concat')
    params = LeakyReLU()(Dense(1024)(params))

    # Apply deconvolution layers

    height, width = initial_shape

    print('height:', height, 'width:', width)

    x = LeakyReLU()(Dense(height * width * num_kernels[0])(params))
    if K.image_dim_ordering() == 'th':
        x = Reshape((num_kernels[0], height, width))(x)
    else:
        x = Reshape((height, width, num_kernels[0]))(x)

    for i in range(0, deconv_layers):
        # Upsample input
        x = UpSampling2D((2, 2))(x)

        # Apply 5x5 and 3x3 convolutions

        # If we didn't specify the number of kernels to use for this many
        # layers, just repeat the last one in the list.
        idx = i if i < len(num_kernels) else -1
        x = LeakyReLU()(Convolution2D(num_kernels[idx], 5, 5, border_mode='same')(x))
        x = LeakyReLU()(Convolution2D(num_kernels[idx], 3, 3, border_mode='same')(x))
        x = BatchNormalization()(x)

    # Last deconvolution layer: Create 3-channel image.
    x = MaxPooling2D((1, 1))(x)
    x = UpSampling2D((2, 2))(x)
    x = LeakyReLU()(Convolution2D(8, 5, 5, border_mode='same')(x))
    x = LeakyReLU()(Convolution2D(8, 3, 3, border_mode='same')(x))
    x = Convolution2D(1 if use_yale else 3, 3, 3, border_mode='same', activation='sigmoid')(x)

    # Compile the model

    if use_yale:
        model = Model(input=[identity_input, pose_input, lighting_input], output=x)
    else:
        model = Model(input=[identity_input, orientation_input, emotion_input], output=x)

    # TODO: Optimizer options
    model.compile(optimizer=optimizer, loss='msle', metrics=[psnr])

    return model


class Generator:

    def __init__(self, model_path, id_len=57, deconv_layer=6):
        model = build_model(
            identity_len=id_len,
            deconv_layers=deconv_layer,
            optimizer='adam',
            initial_shape=(5, 4),
        )
        model.load_weights(model_path)
        self.id_len = id_len
        self.model = model

    def generate(self, id, emo='happy', orient='front', _debug=False):

        if orient == 'front':
            orientation = np.zeros((1, 2))
        else:
            raise NotImplementedError
            
        id_weights = np_utils.to_categorical([id], self.id_len)
        id_weights[:, 10] = 1.0 # testing generation from multiple IDs
            
        input_vec = {
            'identity': id_weights,  # np_utils.to_categorical([id], self.id_len),
            'emotion': np.array(getattr(Emotion, emo)).reshape((1, Emotion.length())),
            'orientation': np.array(orientation),
        }

        gen = self.model.predict_on_batch(input_vec)[0]
        if K.image_dim_ordering() == 'th':
            image = np.empty(gen.shape[2:] + (3,))
            for x in range(0, 3):
                image[:, :, x] = gen[x, :, :]
        else:
            image = gen
        image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
        return image

    def __str__(self):
        return "{}".format(self.__class__.__name__)


if __name__ == '__main__':
    gen = Generator('./output/FaceGen.RaFD.model.d6.adam.iter500.h5')
    for i in range(5):
        image = gen.generate(i, 'happy')
        scipy.misc.imsave('../out/gen_out_' + str(i) + '.jpg', image)
