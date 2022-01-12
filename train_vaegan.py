import sys
import pickle
from args import parse_args, set_seed
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from dataloaders.dataset import load_files
from vaegan_utils.models import create_models, build_graph
from vaegan_utils.training import fit_models
from vaegan_utils.dataloader import encoder_loader, decoder_loader, discriminator_loader, ImgIterator, load_images
from vaegan_utils.callbacks import DecoderSnapshot, ModelsCheckpoint
from dataloaders.ColonCancerDataset import ColonCancerDataset

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def main():
    args = parse_args()
    encoder, decoder, discriminator = create_models()

    encoder_train, decoder_train, discriminator_train, vae, vaegan = build_graph(encoder, decoder, discriminator)

    try:
        initial_epoch = int(sys.argv[1])
    except (IndexError, ValueError):
        initial_epoch = 0

    epoch_format = '.{epoch:03d}.h5'

    if initial_epoch != 0:
        suffix = epoch_format.format(epoch=initial_epoch)
        encoder.load_weights('encoder' + suffix)
        decoder.load_weights('decoder' + suffix)
        discriminator.load_weights('discriminator' + suffix)

    real_acc = tf.keras.metrics.BinaryAccuracy(name="real_acc")
    fake_zp_acc = tf.keras.metrics.BinaryAccuracy(name="fake_z_acc")
    fake_z_acc = tf.keras.metrics.BinaryAccuracy(name="fake_zp_acc")

    rmsprop = RMSprop(lr=0.0003)
    set_trainable(encoder, False)
    set_trainable(decoder, False)
    discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, [real_acc,fake_z_acc, fake_zp_acc])
    discriminator_train.summary()


    decoder_z_acc = tf.keras.metrics.BinaryAccuracy(name="decoder_z_acc")
    decoder_zp_acc = tf.keras.metrics.BinaryAccuracy(name="decoder_zp_acc")

    set_trainable(discriminator, False)
    set_trainable(decoder, True)
    decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2,[decoder_z_acc,decoder_zp_acc])
    decoder_train.summary()

    set_trainable(decoder, False)
    set_trainable(encoder, True)
    encoder_train.compile(rmsprop)
    encoder_train.summary()

    set_trainable(vaegan, True)

    checkpoint = ModelsCheckpoint("vaegan_weights/",epoch_format, encoder, decoder, discriminator)
    decoder_sampler = DecoderSnapshot("decode_dir/")

    callbacks = [checkpoint, decoder_sampler]

    epochs = 70

    seed = np.random.randint(2**32 - 1)

    model_train_set = ColonCancerDataset(patch_size=27, augmentation=False).load_bags(train_bags)

    batch_size=4
    hdf5Iterator = ImgIterator(model_train_set, batch_size=batch_size, shuffle=True)
    steps_per_epoch=len(hdf5Iterator)
    img_loader = load_images(hdf5Iterator,num_child=3)

    dis_loader = discriminator_loader(img_loader, seed=seed)
    dec_loader = decoder_loader(img_loader, seed=seed)
    enc_loader = encoder_loader(img_loader)

    models = [discriminator_train, decoder_train, encoder_train]

    generators = [dis_loader, dec_loader, enc_loader]

    histories = fit_models(vaegan, models, generators, batch_size=batch_size,
                           steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                           epochs=epochs, initial_epoch=initial_epoch)


if __name__ == '__main__':
    main()
