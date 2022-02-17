from __future__ import absolute_import
from __future__ import print_function
from tensorflow.keras.preprocessing.image import Iterator
import numpy as np
from dataloaders.data_aug_op import  random_flip_img,random_rotate_img

class ImgIterator(Iterator):
    def __init__(self,
                 train_set,
                 batch_size,
                 directory=".",
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 color_mode='rgb',
                 target_size=(27,27)):

        self.color_mode = color_mode
        self.directory = directory
        self.train_set=train_set
        self.current_filename_index=0
        self.data_format = data_format
        self.target_size = target_size
        self.shuffle=shuffle
        self.batch_size=batch_size

        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.aug_batch = []
        for ibatch, batch in enumerate(train_set):
            img_data = batch[0]
            for i in range(img_data.shape[0]):
                ori_img = img_data[i, :, :, :]

                if shuffle:
                    img = random_flip_img(ori_img, horizontal_chance=0.5, vertical_chance=0.5)
                    img = random_rotate_img(img)

                else:
                    img = ori_img
                self.aug_batch.append(img)

        super(ImgIterator, self).__init__(len(self.aug_batch), self.batch_size, shuffle, seed)


    def _get_batches_of_transformed_samples(self, index_array):


        batch_x = self.aug_batch[index_array[0]:index_array[0] + index_array.shape[0]]
        batch_x = np.array(batch_x, dtype="float32")

        return batch_x

    def getitem(self, index):
        return self.__getitem__(index)

def load_images(iterator,num_child=4):

    while True:
        for batch_images in iterator:
            for i in range(num_child):
                yield batch_images

def discriminator_loader(img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))

        y_real = np.ones((batch_size,), dtype='float32')
        y_fake = np.zeros((batch_size,), dtype='float32')

        yield [x, z_p], [y_real, y_fake, y_fake]

def decoder_loader(img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))
        # Label as real
        y_real = np.ones((batch_size,), dtype='float32')
        yield [x, z_p], [y_real, y_real]

def encoder_loader(img_loader):
    while True:
        x = next(img_loader)
        yield x, None
