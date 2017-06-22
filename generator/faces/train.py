"""
faces/train.py
"""

import os

from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint #LearningRateScheduler, BaseLogger
#from keras.schedules import TriangularLearningRate
from keras.models import load_model

import numpy as np
import scipy.misc

from .instance import Emotion, RaFDInstances, YaleInstances, NUM_YALE_POSES
from .model import build_model


class LrReducer(Callback):
    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_acc')
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= 10:
                    lr = self.model.optimizer.lr.get_value()
                    self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1

class GenerateIntermediate(Callback):
    """ Callback to generate intermediate images after each epoch. """

    def __init__(self, output_dir, num_identities, batch_size=32, use_yale=False):
        """
        Constructor for a GenerateIntermediate object.

        Args:
            output_dir (str): Directory to save intermediate results in.
            num_identities (int): Number of identities in the training set.
        Args: (optional)
            batch_size (int): Batch size to use when generating images.
        """
        super(Callback, self).__init__()

        self.output_dir = output_dir
        self.num_identities = num_identities
        self.batch_size = batch_size
        self.use_yale = use_yale

        self.parameters = dict()

        # Sweep through identities
        self.parameters['identity'] = np.eye(num_identities)

        if use_yale:
            # Use pose 0, lighting at 0deg azimuth and elevation
            self.parameters['pose'] = np.zeros((num_identities, NUM_YALE_POSES))
            self.parameters['lighting'] = np.zeros((num_identities, 4))
            for i in range(0, num_identities):
                self.parameters['pose'][i,0] = 0
                self.parameters['lighting'][i,1] = 1
                self.parameters['lighting'][i,3] = 1
        else:
            # Make all have neutral expressions, front-facing
            self.parameters['emotion'] = np.empty((num_identities, Emotion.length()))
            self.parameters['orientation'] = np.zeros((num_identities, 2))
            for i in range(0, num_identities):
                self.parameters['emotion'][i,:] = Emotion.neutral
                self.parameters['orientation'][i,1] = 1


    def on_train_begin(self, logs={}):
        """ Create directories. """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def on_epoch_end(self, epoch, logs={}):
        """ Generate and save results to the output directory. """

        dest_dir = os.path.join(self.output_dir, 'e{:04}'.format(epoch))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        gen = self.model.predict(self.parameters, batch_size=self.batch_size)

        for i in range(0, gen.shape[0]):
            if K.image_dim_ordering() == 'th':
                if self.use_yale:
                    image = np.empty(gen.shape[2:])
                    image[:,:] = gen[i,0,:,:]
                else:
                    image = np.empty(gen.shape[2:]+(3,))
                    for x in range(0, 3):
                        image[:,:,x] = gen[i,x,:,:]
            else:
                if self.use_yale:
                    image = gen[i,:,:,0]
                else:
                    image = gen[i,:,:,:]
            image = np.array(255*np.clip(image,0,1), dtype=np.uint8)
            file_path = os.path.join(dest_dir, '{:02}.png'.format(i))
            scipy.misc.imsave(file_path, image)    

def train_model(data_dir, output_dir, model_file='', batch_size=32,
        num_epochs=100, optimizer='adam', deconv_layers=5, use_yale=False,
        kernels_per_layer=None, generate_intermediate=False, verbose=False):
    """
    Trains the model on the data, generating intermediate results every epoch.

    Args:
        data_dir (str): Directory where the data lives.
        output_dir (str): Directory where outputs should be saved.
        model_file (str): Model file to load. If none specified, a new model
            will be created.
    Args (optional):
        batch_size (int): Size of the batch to use.
        num_epochs (int): Number of epochs to train for.
        optimizer (str): Keras optimizer to use.
        deconv_layers (int): The number of deconvolution layers to use.
        generate_intermediate (bool): Whether or not to generate intermediate results.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    instances = YaleInstances(data_dir) if use_yale else RaFDInstances(data_dir)

    if verbose:
        print("Found {} instances with {} identities".format(
            instances.num_instances, instances.num_identities))


    # Create FaceGen model to use

    if model_file:
        model = load_model(model_file)
        if verbose:
            print("Loaded model from {}".format(model_file))
    else:
        model = build_model(
            identity_len  = instances.num_identities,
            deconv_layers = deconv_layers,
            num_kernels   = kernels_per_layer,
            optimizer     = optimizer,
            initial_shape = (5,4) if not use_yale else (6,8),
            use_yale      = use_yale,
        )
        if verbose:
            print("Built model with:")
            print("\tDeconv layers: {}".format(deconv_layers))
            print("\tOutput shape: {}".format(model.output_shape[1:]))


    # Create training callbacks

    callbacks = list()

    if generate_intermediate:
        intermediate_dir = os.path.join(output_dir, 'intermediate.d{}.{}'.format(deconv_layers, optimizer))
        callbacks.append( GenerateIntermediate(intermediate_dir, instances.num_identities, use_yale=use_yale) )

    model_path = os.path.join(output_dir, 'FaceGen.{}.model.d{}.{}.h5'
            .format('YaleFaces' if use_yale else 'RaFD', deconv_layers, optimizer))

    callbacks.append(
        ModelCheckpoint(
            model_path,
            monitor='loss', verbose=0, save_best_only=True,
        )
    )
    #callbacks.append(
    #    EarlyStopping(monitor='loss', patience=16)
    #)

    # Load data and begin training

    if verbose:
        print("Loading data...")

    if K.image_dim_ordering() == 'th':
        image_size = model.output_shape[2:4]
    else:
        image_size = model.output_shape[1:3]

    inputs, outputs = instances.load_data(image_size, verbose=verbose)

    if verbose:
        print("Training...")

    #curr_loss 
    #while num_epochs > 0:
    # loop one by one?
    historyObj = model.fit(inputs, outputs, batch_size=batch_size, nb_epoch=num_epochs,
            callbacks=callbacks, shuffle=True, verbose=1)
    print("History: \n")
    print(historyObj.history)


    if verbose:
        print("Done!")

