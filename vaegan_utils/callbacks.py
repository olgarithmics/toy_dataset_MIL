#!/usr/bin/env python3

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

class DecoderSnapshot(Callback):

    def __init__(self, decode_dir, step_size=100, latent_dim=128, decoder_index=-2):
        super().__init__()
        self._step_size = step_size
        self._steps = 0
        self._epoch = 0
        self._latent_dim = latent_dim
        self._decoder_index = decoder_index
        self._img_rows = 27
        self._img_cols = 27
        self.decode_dir=decode_dir
        self._thread_pool = ThreadPoolExecutor(1)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._steps = 0

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._steps % self._step_size == 0:
            self.sample_images()


    def sample_images(self):
        r, c = 4, 4
        decoder = self.model.layers[self._decoder_index]
        filename = self.decode_dir + 'generated_%d_%d.png' % (self._epoch, self._steps)
        fig, axs = plt.subplots(r, c)
        for i in range(c):
            z = np.random.normal(size=(c, self._latent_dim))
            images = decoder.predict(z)
            images = (images + 1.) * 127.5
            images = np.clip(images, 0., 255.)
            images = images.astype('uint8')
            for j in range(r):
                axs[j,i].imshow(images[j,:,:,:])
                axs[j,i].axis('off')
        fig.savefig(filename)
        plt.close()


    def plot_images(self, samples=4):
        decoder = self.model.layers[self._decoder_index]
        filename = self.decode_dir+'generated_%d_%d.png' % (self._epoch, self._steps)
        z = np.random.normal(size=(samples, self._latent_dim))
        images = decoder.predict(z)
        self._thread_pool.submit(self.save_plot, images, filename)

    @staticmethod
    def save_plot(images, filename):
        images = (images + 1.) * 127.5
        images = np.clip(images, 0., 255.)
        images = images.astype('uint8')
        rows = []
        for i in range(0, len(images), 2):
            rows.append(np.concatenate(images[i:(i + 2), :, :, :], axis=0))
        plot = np.concatenate(rows, axis=1).squeeze()
        Image.fromarray(plot).save(filename)


class ModelsCheckpoint(Callback):

    def __init__(self, weight_dir,epoch_format,*models):
        super().__init__()
        self._epoch_format = epoch_format
        self._models = models
        self.weight_dir=weight_dir

    def on_epoch_end(self, epoch, logs=None):
        suffix = self._epoch_format.format(epoch=epoch + 1, **logs)
        for model in self._models:

            model.save_weights(self.weight_dir+model.name + suffix)
