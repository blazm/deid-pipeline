
from keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

from Generator import Generator, build_model as build_generator, psnr, log10
from Discriminator import build_model as build_discriminator
from faces.instance import YaleInstances, RaFDInstances

import numpy as np
from time import time, strftime

import tensorflow as tf 

#
#writer.add_run_metadata(tf.RunMetadata(), "demo_run", global_step=None)

# exec tensorboard via python: python -m tensorflow.tensorboard --logdir=

def log(tag, value):
    writer = tf.summary.FileWriter("./board")
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, 
                                     simple_value=value), ])
    writer.add_summary(summary)


def print_weights(model):



    weights = model.trainable_weights # weight tensors
    weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
    gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors


    for grad in gradients:
        g = tf.Variable(tf.truncated_normal(g))
        g_summary = tf.image_summary(g)


    #print(weights)
    # ==> [dense_1_W, dense_1_b]

#def build_model(id_len=57, deconv_layer=6, initial_shape=(5, 4)):
#    pass

def train():
    pass


    images_train = self.x_train[np.random.randint(0,
    self.x_train.shape[0], size=batch_size), :, :, :]
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    images_fake = self.generator.predict(noise)
    x = np.concatenate((images_train, images_fake))
    y = np.ones([2*batch_size, 1])
    y[batch_size:, :] = 0
    d_loss = self.discriminator.train_on_batch(x, y)
    y = np.ones([batch_size, 1])
    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    a_loss = self.adversarial.train_on_batch(noise, y)

def max_log_loss(y_true, y_pred):
    return K.sum(K.maximum(K.log(y_pred)))


if __name__ == '__main__':

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1" # in the external slot
    #os.environ["CUDA_VISIBLE_DEVICES"]="0" # on the mobo

    experiment_label = "lr=0.001"

    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.iter500.h5')
    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.h5')
    #emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    id_len = 57
    deconv_layer = 2
    num_epochs = 50
    batch_size = 64
    #gen_path = './output/FaceGen.RaFD.model.d2.adam.h5'
    #gen_model = build_generator(identity_len=id_len, deconv_layers=deconv_layer, optimizer='adam', initial_shape=(5, 4))
    #gen_model.load_weights(gen_path)
    #gen_model.summary()


    # fake images
    
    # TODO: generate from random picked IDs (similar to random uniform picking from latent space)
    #gen = Generator(None, deconv_layer=deconv_layer) # try to train empty Generator
    gen = Generator('./output/FaceGen.RaFD.model.d{}.adam.h5'.format(deconv_layer), deconv_layer=deconv_layer)
    gen_model = gen.getKerasModel()
    #gen_model.summary()

    dis_path = './output/FaceDisc.RaFD.model.c{}.adam.h5'.format(deconv_layer)
    dis_model = build_discriminator(optimizer='adam', initial_shape=(5, 4), conv_layers=deconv_layer)
    #dis_model.load_weights(dis_path)
    #dis_model.summary()

    dis_model.trainable = False # does this work?
    adv_model = Sequential()
    adv_model.add(gen_model)
    adv_model.add(dis_model)

    adv_model.summary()

    #optimizer = RMSprop(lr=0.00005, clipvalue=1.0, decay=3e-8)
    #optimizer = RMSprop(lr=0.01, clipvalue=1.0, decay=3e-8)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adv_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # , psnr

    #return adv_model

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


    #gen_imgs = gen.generate_actual()
    # gen_imgs = []
    # for emotion in emotions:    
    #     for i in range(1, 57): # 3x k=2
    #         # k=2
    #         ids = [i, i-1]; # , i-1
    #         image = gen.generate(i, emotion)
    #         gen_imgs.append(image)

    # gen_imgs = np.array(gen_imgs)
    # print("FAKE INPUTS: ", gen_imgs.shape)

    # real images
    input_params, real_imgs = instances.load_data(image_size, verbose=verbose)
    #print(inputs.keys()) # identities
    #print(outputs.shape) # images
  

    print("MAX MIN REAL: ", np.max(real_imgs[0]), np.min(real_imgs[0]))


    # balance number of generated images with number of real images
    # generating 1k results in non-zero predictions (nonbalanced dataset?)
    gen_imgs = gen.generate_random(real_imgs.shape[0]) 
    gen_imgs = gen.generate_actual('../out/before_adversarial/') 
    
    gen_imgs = gen_imgs / 255.0
    print("MAX MIN GENR: ", np.max(gen_imgs[0]), np.min(gen_imgs[0]))
    
    #from scipy.misc import toimage
    #toimage(gen_imgs[1]).show(title='generated')
    #from scipy.misc import toimage
    #toimage(real_imgs[1]).show(title='real')
      
  
    # TODO: merge inputs and fake inputs before training 

    inputs = real_imgs
    outputs = np.ones((real_imgs.shape[0], 1))

    #inputs = np.concatenate((gen_imgs, real_imgs)) # [iimgs]
    #outputs = np.concatenate((np.zeros((gen_imgs.shape[0], 1)), np.ones((real_imgs.shape[0], 1))), axis=0) # [1, 0]

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    #inputs, outputs = unison_shuffled_copies(inputs, outputs)

    num_seq = inputs.shape[0]

    print(inputs.shape)
    print(outputs.shape)


    def train_gen():
        while True:
            for i in range(0, num_seq + (num_seq % batch_size),batch_size):
                                
                X,y = inputs[i:i+batch_size], outputs[i:i+batch_size] #get_sequences(length=i)
                yield X, y


    if verbose:
        print("Training...")


    #board = TensorBoard(log_dir='./board/{}'.format(time()), histogram_freq=0,  
    #      write_graph=True, write_images=True)

    callbacks_d = list() # TODO: generate immediate predictions after each epoch*
    callbacks_a = list() # TODO: generate immediate predictions after each epoch*
    callbacks_g = list() # TODO: generate immediate predictions after each epoch*

    output_dir = 'output/'
    optimizer = 'adam'

    model_path = os.path.join(output_dir, 'FaceAdv.{}.model.c{}.{}.h5'
            .format('YaleFaces' if use_yale else 'RaFD', deconv_layer, optimizer))

    callbacks_a.append(
        ModelCheckpoint(
            model_path,
            monitor='loss', verbose=0, save_best_only=True,
        )
    )

    model_path_g = os.path.join(output_dir, 'AdvFaceGen.{}.model.d{}.{}.h5'
                .format('YaleFaces' if use_yale else 'RaFD', deconv_layer, optimizer))

    callbacks_g.append(
        ModelCheckpoint(
            model_path_g,
            monitor='loss', verbose=0, save_best_only=True,
        )
    )

    model_path = os.path.join(output_dir, 'FaceDisc.{}.model.c{}.{}.h5'
                .format('YaleFaces' if use_yale else 'RaFD', deconv_layer, optimizer))

    callbacks_d.append(
        ModelCheckpoint(
            model_path,
            monitor='loss', verbose=0, save_best_only=True,
        )
    )

    data_flow = train_gen()

    tensorboard = TensorBoard(log_dir='./board/{}_{}'.format(strftime("%Y-%m-%d_%H.%M"), experiment_label), histogram_freq=0,
                          write_graph=True, write_images=True, write_grads=True, 
                          batch_size=batch_size)
    #tensorboard.set_params(adv_model.get_config())
    tensorboard.set_model(adv_model)
    #print(input_params)
    #print(outputs)

    tensors = (adv_model.inputs +
                           adv_model.targets +
                           adv_model.sample_weights)
    print(len(tensors))
    print(len((input_params, outputs)))

    print(len(input_params))

    a, b, c = input_params

    tensorboard.validation_data = (a, b, c, outputs, None)

 #   print(adv_model.get_config())

    for e in range(num_epochs):
        print("EPOCHS: ", e, "/", num_epochs)

       # tensorboard.on_epoch_begin(e)

        for i in range(0, num_seq // batch_size):
            #images_train = self.x_train[np.random.randint(0,
            #    self.x_train.shape[0], size=batch_size), :, :, :]

          #  tensorboard.on_batch_begin(i)

            images_train, labels_real = next(data_flow)

            #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            #noise = gen.generate_random_inputs(batch_size)
            images_fake = gen.generate_random(batch_size) # generator predict
            images_fake = images_fake / 255.0
    
            #from scipy.misc import toimage
            #toimage(images_train[1]).show(title='generated')

            #x = np.concatenate((images_train, images_fake))
            #y = np.ones([2*batch_size, 1])
            #y[batch_size:, :] = 0


            #x = np.concatenate((images_fake, images_train)) # [iimgs]
            #y = np.concatenate((np.zeros((images_fake.shape[0], 1)), 
            #                    np.ones((images_train.shape[0], 1))), axis=0) # [1, 0]

           # x, y = unison_shuffled_copies(x, y) # should not shuffle batches
            #print(x.shape)
            #print(y.shape)

            #train separately
            x = images_fake
#            y = np.zeros([images_fake.shape[0], 1])
            y = np.zeros([images_fake.shape[0], 1]) \
            #    + np.random.normal(loc=0.0, scale=0.03, size=[images_fake.shape[0], 1])
            #print("MAX MIN dfY: ", np.max(y), np.min(y))
    
            d_loss = dis_model.train_on_batch(x, y) # discriminator train

           # log("d_loss_fake", d_loss[0])
           # log("d_acc_fake", d_loss[1])

            x = images_train
 #           y = np.ones([images_train.shape[0], 1])
            y = np.ones([images_train.shape[0], 1]) \
            #    + np.random.normal(loc=0.0, scale=0.03, size=[images_train.shape[0], 1])
            #print("MAX MIN drY: ", np.max(y), np.min(y))
    
            d_loss = dis_model.train_on_batch(x, y) # discriminator train

           # log("d_loss_real", d_loss[0])
           # log("d_acc_real", d_loss[1])

            y = np.ones([batch_size, 1]) \
            #    + np.random.normal(loc=0.0, scale=0.03, size=[batch_size, 1])

            #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            input_vec = gen.generate_random_inputs(batch_size)

            #print(input_vec['identity'].shape)
            #print(input_vec['emotion'].shape)
            #print(input_vec['orientation'].shape)

            #print(input_vec[2])

            #print(y.shape)

            a_loss = adv_model.train_on_batch(input_vec, y) # adversarial train
            #callbacks = None
            #a_loss = adv_model.fit(input_vec, y, batch_size=batch_size, nb_epoch=num_epochs,
            #callbacks=callbacks, shuffle=True, verbose=1)

            log_mesg = "\r%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            logs = {}
            logs["a_loss"] = np.float32(a_loss[0])
            logs["d_loss"] = np.float32(d_loss[0])

            logs["a_acc"] = np.float32(a_loss[1])
            logs["d_acc"] = np.float32(d_loss[1])
         #   tensorboard.on_batch_end(i, logs=logs)

#            print_weights(gen_model)

           # log("a_loss", a_loss[0])
           # log("a_acc", a_loss[1])


            tensorboard.on_epoch_end(e * num_seq // batch_size + i, logs=logs)


        gen_model.save(model_path_g)
        #gen.generate_random(10, None,'../out/adversarial/')
        gen.generate_actual('../out/adversarial/')

    tensorboard.on_train_end(None)
    #curr_loss 
    #while num_epochs > 0:
    # custom training using generator


    

    # historyObj = model.fit_generator(train_gen, steps_per_epoch=(num_seq // batch_size)+1, 
    #         epochs=num_epochs, callbacks=callbacks)

    # # loop one by one?
    # #historyObj = model.fit(inputs, outputs, batch_size=batch_size, nb_epoch=num_epochs,
    #         callbacks=callbacks, shuffle=True, verbose=1)
    # print("History: \n")
    # print(historyObj.history)


    if verbose:
        print("Done!")