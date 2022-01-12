import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation,GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.resnet50 import ResNet50
from .losses import mean_gaussian_negative_log_likelihood


def create_models(n_channels=3, recon_depth=2, wdecay=1e-5):

    image_shape = (27, 27, n_channels)
    n_discriminator = 256
    latent_dim = 256
    decode_from_shape = (6, 6, 64)
    n_decoder = np.prod(decode_from_shape)

    # Encoder
    def create_encoder():

        inputs = Input(shape=image_shape, name='enc_input')
        x = Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu")(inputs)
        x = Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu")(x)
        x = Flatten()(x)
        x = Dense(latent_dim, name='x_mean')(x)

        z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='he_uniform')(x)
        z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='he_uniform')(x)

        return Model(inputs, [z_mean, z_log_var], name='encoder')

    decoder = Sequential([
        Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=(latent_dim,),
              name='dec_h_dense'),
        Reshape(decode_from_shape),
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid'),
        Conv2D(3, 3, activation='tanh', padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform',
               name='dec_output')
    ], name='decoder')

    def create_discriminator():
        x = Input(shape=image_shape, name='dis_input')

        layers = [
            Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='valid'),

            Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='valid'),

            Flatten(),
            Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_dense'),

            Dense(1, activation='sigmoid', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_output')
        ]

        y = x
        y_feat = None
        for i, layer in enumerate(layers, 1):
            y = layer(y)
            # Output the features at the specified depth
            if i == recon_depth:
                y_feat = y
        return Model(x, [y, y_feat], name='discriminator')

    encoder = create_encoder()
    discriminator = create_discriminator()

    return encoder, decoder, discriminator


def _sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       Instead of sampling from Q(z|X), sample eps = N(0,I)

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_graph(encoder, decoder, discriminator, recon_vs_gan_weight=1e-6):

    image_shape = K.int_shape(encoder.input)[1:]

    latent_shape = K.int_shape(decoder.input)[1:]

    sampler = Lambda(_sampling, output_shape=latent_shape, name='sampler')

    x = Input(shape=image_shape, name='input_image')

    z_p = Input(shape=latent_shape, name='z_p')

    z_mean, z_log_var = encoder(x)
    z = sampler([z_mean, z_log_var])

    x_tilde = decoder(z)
    x_p = decoder(z_p)


    dis_x, dis_feat = discriminator(x)
    dis_x_tilde, dis_feat_tilde = discriminator(x_tilde)
    dis_x_p = discriminator(x_p)[0]

    # Learned similarity metric
    dis_nll_loss = mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)

    # KL divergence loss
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))

    # Create models for training
    encoder_train = Model(x, dis_feat_tilde, name='e')
    encoder_train.add_loss(kl_loss)
    encoder_train.add_loss(dis_nll_loss)

    decoder_train = Model([x, z_p], [dis_x_tilde, dis_x_p], name='de')

    normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)
    decoder_train.add_loss(normalized_weight * dis_nll_loss)

    discriminator_train = Model([x, z_p], [dis_x, dis_x_tilde, dis_x_p], name='di')


    vae = Model(x, x_tilde, name='vae')
    vaegan = Model(x, dis_x_tilde, name='vaegan_utils')

    return encoder_train, decoder_train, discriminator_train, vae, vaegan
