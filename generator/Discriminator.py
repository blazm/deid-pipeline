import scipy
from keras import backend as K
import numpy as np
from keras.utils import np_utils
from scipy import misc
from keras.layers import BatchNormalization, Convolution2D, Conv2D, Dense, LeakyReLU, \
    Input, MaxPooling2D, merge, Reshape, Dropout, Flatten
from keras.models import Model
import tensorflow as tf

from Generator import Generator, psnr, log10

from faces.instance import YaleInstances, RaFDInstances

def build_model(initial_shape=(5, 4), conv_layers=5, 
                num_kernels=None, optimizer='adam'):
    """
    Builds a convolution Discriminator model.

    Args (optional):
        
        initial_shape (tuple<int>): The starting shape of the deconv. network.
        conv_layers (int): How many conv. layers to use (same as deconv layers in Generator?). 
            More layers gives better resolution, although requires more GPU memory.
        num_kernels (list<int>): Number of convolution kernels for each layer.
        optimizer (str): The optimizer to use. Will only use default values.
    Returns:
        keras.Model, the constructed model.
    """

    if num_kernels is None:
        num_kernels = [128, 128, 96, 96, 32, 32, 16]
        num_kernels.reverse()

    # generator output - initial * 2 (upsampling by 2) ** (deconv + last layer)    
    height, width = initial_shape
    input_shape = (height*2**(conv_layers+1), width*2**(conv_layers+1), 3)

    print(input_shape)

    image_input = Input(shape=input_shape, name='image')
    x = image_input

    for i in range(0, conv_layers+1): # do we need +1?
        idx = i if i < len(num_kernels) else -1

        x = Conv2D(num_kernels[idx], 5, strides=2, input_shape=input_shape, \
            padding='same', activation=LeakyReLU(alpha=0.2), name='conv_'+str(i+1))(x)
        x = Dropout(rate=0.3)(x)
        x = BatchNormalization()(x)

    x = Flatten()(x)
    # TODO: include more dense layers
    x = LeakyReLU()(Dense(512)(x))
    x = LeakyReLU()(Dense(256)(x))
    x = LeakyReLU()(Dense(128)(x))
    # if needed multiple outputs (also guessing identity)
    #out_id = Dense(57, activation='softmax', name='auxiliary')(x) 
    x = Dense(1, activation='sigmoid', name='guess')(x)


    model = Model(input=image_input, output=x)

    # TODO: Optimizer options
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[psnr, 'accuracy']) # TODO: will this affect adversarial model compiling?

    model.summary()

    return model

class Discriminator:

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

    def classify(self, image, _debug=False):
        result = self.model.predict_on_batch(image)
        print(result)
        # if orient == 'front':
        #     orientation = np.zeros((1, 2))
        # else:
        #     raise NotImplementedError
            
        # if type(id) is list:
        #     id_weights = np_utils.to_categorical([id[1]], self.id_len)
        #     for i in id:
        #         id_weights[:, i] = 1.0
        #         id_weights = id_weights / (1.0 * len(id));
        #     #id_weights = np_utils.to_categorical(id, self.id_len)
        # else:
        #     id_weights = np_utils.to_categorical([id], self.id_len)
            
        # #id_weights[:, 10] = 1.0 # testing generation from multiple IDs
            
        # input_vec = {
        #     'identity': id_weights,  # np_utils.to_categorical([id], self.id_len),
        #     'emotion': np.array(getattr(Emotion, emo)).reshape((1, Emotion.length())),
        #     'orientation': np.array(orientation),
        # }

        # gen = self.model.predict_on_batch(input_vec)[0]
        # if K.image_dim_ordering() == 'th':
        #     image = np.empty(gen.shape[2:] + (3,))
        #     for x in range(0, 3):
        #         image[:, :, x] = gen[x, :, :]
        # else:
        #     image = gen
        # image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
        # return image

    def __str__(self):
        return "{}".format(self.__class__.__name__)



if __name__ == '__main__':
    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.iter500.h5')
    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.h5')
    #emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
    #id_len = 57
    #deconv_layer = 2

    #gen_model = build_generator(identity_len=id_len, deconv_layers=deconv_layer, optimizer='adam', initial_shape=(5, 4))
    #gen_model.load_weights(gen_path)
    #gen_model.summary()


    dis_model = build_model(
        optimizer='adam',
        initial_shape=(5, 4)
    )
    #model.load_weights(model_path)

    # verify if Discriminator model is correct by training it alone with real and fake images

    data_dir = '../DB/rafd2-frontal/' # TODO: param
    verbose = True
    use_yale = False
    # training
    instances = YaleInstances(data_dir) if use_yale else RaFDInstances(data_dir)


    if verbose:
        print("Loading data...")

    if K.image_dim_ordering() == 'th':
        image_size = dis_model.input_shape[2:4] # or gen_model.output_shape
    else:
        image_size = dis_model.input_shape[1:3]


    # fake images
    gen = Generator('./output/FaceGen.RaFD.model.d5.adam.h5', deconv_layer=5)
    emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    
    gen_imgs = []
    for emotion in emotions:    
        for i in range(1, 57): # 3x k=2
            # k=2
            ids = [i, i-1]; # , i-1
            image = gen.generate(i, emotion)
            gen_imgs.append(image)

    gen_imgs = np.array(gen_imgs)
    print("FAKE INPUTS: ", gen_imgs.shape)


    # real images
    input_params, real_imgs = instances.load_data(image_size, verbose=verbose)

    #print(inputs.keys()) # identities
    #print(outputs.shape) # images

  
    # TODO: merge inputs and fake inputs before training 
    inputs = np.concatenate((gen_imgs, real_imgs), axis=0) # [iimgs]
    outputs = np.concatenate((np.zeros((gen_imgs.shape[0], 1)), np.ones((real_imgs.shape[0], 1))), axis=0) # [1, 0]

    print(inputs.shape)
    print(outputs.shape)

    if verbose:
        print("Training...")

    #curr_loss 
    #while num_epochs > 0:
    # loop one by one?

    batch_size = 16
    num_epochs = 10
    callbacks = list() # TODO: generate immediate predictions after each epoch*

    historyObj = dis_model.fit(inputs, outputs, batch_size=batch_size, nb_epoch=num_epochs,
            callbacks=callbacks, shuffle=True, verbose=1)
    print("History: \n")
    print(historyObj.history)


    gen_imgs = []
    for emotion in emotions:    
        for i in range(1, 57): # 3x k=2
            # k=2
            ids = [i, i-1]; # , i-1
            image = gen.generate(ids, emotion)
            gen_imgs.append(image)

    gen_imgs = np.array(gen_imgs)

    results = dis_model.predict_on_batch(gen_imgs)
    print(results)


    if verbose:
        print("Done!")