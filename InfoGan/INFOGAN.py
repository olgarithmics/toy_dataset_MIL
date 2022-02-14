from __future__ import print_function, division
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape,Dropout ,\
    Lambda, LeakyReLU, Activation,GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks as cbks
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

class INFOGAN():
    def __init__(self):
        self.img_rows = 27
        self.img_cols = 27
        self.channels = 3
        self.num_classes = 4
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128
        self.decode_from_shape = (6, 6, 512)
        self.n_decoder = np.prod(self.decode_from_shape)
        self.wdecay = 1e-5
        self.leaky_relu_alpha=0.2
        self.n_discriminator=512


        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', self.mutual_info_loss]

        # Build and the discriminator and recognition network
        self.discriminator, self.auxilliary = self.build_disk_and_q_net()

        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the recognition network Q
        self.auxilliary.compile(loss=[self.mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        gen_input = Input(shape=(self.latent_dim,))
        img = self.generator(gen_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        valid = self.discriminator(img)
        # The recognition network produces the label
        target_label = self.auxilliary(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(gen_input, [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):

        generator = Sequential([
            Dense(self.n_decoder, kernel_regularizer=l2(self.wdecay), kernel_initializer='he_uniform',
                  input_shape=(self.latent_dim,),
                  name='dec_h_dense',activation='relu'),
            Reshape(self.decode_from_shape),

            Conv2D(128, (4, 4), padding='same',activation="relu"),

            Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2),activation="relu"),

            Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),activation="relu"),

            Conv2D(3, 3, strides=(1, 1), activation='tanh', padding='same', kernel_regularizer=l2(self.wdecay),
                   kernel_initializer='he_uniform',
                   name='dec_output')
        ], name='decoder')

        return generator


    def build_disk_and_q_net(self):

        img = Input(shape=self.img_shape)

        # Shared layers between discriminator and recognition network
        model = Sequential()
        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(LeakyReLU(alpha=0.2))


        model.add(Flatten())

        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)
        # Recognition
        q_net = Dense(self.n_discriminator, activation='relu', name="feature_descriptor")(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)
        # Return discriminator and recognition network
        return Model(img, validity), Model(img, label)

    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy

    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 124))
        sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        return sampled_noise, sampled_labels

    def train(self,generator, epochs,irun, ifold, sample_interval=50):


        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images

            imgs = next(generator)
            valid = np.ones((len(imgs), 1))
            fake = np.zeros((len(imgs), 1))

            # Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(len(imgs))
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)


            # Generate a half batch of new images
            gen_imgs = self.generator.predict(gen_input)

            # Train on real and generated data
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            # Avg. loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and Q-network
            # ---------------------

            g_loss = self.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # Plot the progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                #self.sample_images(epoch)
                self.save_model(irun, ifold)

    def sample_images(self, epoch):
        r, c = 4, 4

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.num_classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            images = (gen_imgs + 1.) * 127.5
            images = np.clip(images, 0., 255.)
            images = images.astype('uint8')

            for j in range(r):
                axs[j,i].imshow(images[j,:,:,:])
                axs[j,i].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()


    def save_model(self, irun, ifold):

        def save(model, model_name):

            weights_path = "infogan_weights/irun{}_ifold{}/{}_weights.hdf5".format(irun, ifold,model_name)


            model.save_weights(weights_path)

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")
        save(self.auxilliary, "auxilliary")



