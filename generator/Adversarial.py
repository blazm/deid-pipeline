
from keras.models import Model, Sequential
from keras.optimizers import RMSprop

from Generator import build_model as build_generator, psnr, log10
from Discriminator import build_model as build_discriminator
    
from faces.instance import YaleInstances, RaFDInstances

def build_model(id_len=57, deconv_layer=6, initial_shape=(5, 4)):
    pass

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


if __name__ == '__main__':
    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.iter500.h5')
    #gen = Generator('./output/FaceGen.RaFD.model.d6.adam.h5')
    #emotions = ['happy','angry', 'contemptuous', 'disgusted', 'fearful', 'neutral', 'sad', 'surprised']
    id_len = 57
    deconv_layer = 2

    gen_model = build_generator(identity_len=id_len, deconv_layers=deconv_layer, optimizer='adam', initial_shape=(5, 4))
    #gen_model.load_weights(gen_path)
    #gen_model.summary()

    dis_model = build_discriminator(optimizer='adam', initial_shape=(5, 4), conv_layers=deconv_layer)
    #dis_model.load_weights(dis_path)
    #dis_model.summary()

    adv_model = Sequential()
    adv_model.add(gen_model)
    adv_model.add(dis_model)

    adv_model.summary()

    optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
    adv_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', psnr])

    #return adv_model

    data_dir = '../DB/rafd2-frontal/' # TODO: param
    verbose = True
    # training
    instances = YaleInstances(data_dir) if use_yale else RaFDInstances(data_dir)


    if verbose:
        print("Loading data...")

    if K.image_dim_ordering() == 'th':
        image_size = gen_model.output_shape[2:4]
    else:
        image_size = gen_model.output_shape[1:3]

    # real images
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