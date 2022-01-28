import random
from scipy.spatial.distance import cdist
import numpy as np
import tensorflow as tf
from sklearn.metrics import euclidean_distances
from dataloaders.dataset import get_coordinates
from dataloaders.data_aug_op import random_flip_img, random_rotate_img
from multiprocessing import pool


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, prob,k, data_set, trained_model=None, mode="euclidean", shuffle=True, batch_size=1):

        self.prob=prob
        self.data_set = data_set
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.trained_model = trained_model
        self.k = k
        self.mode = mode
        thread_pool = pool.ThreadPool()
        # self.bag_batch, self.neighbors, self.bag_label = self.__data_generation(data_set)
        self.bags, self.neighbors, self.bag_label = zip(*thread_pool.map(self.__data_generation, data_set))

        thread_pool.close()
        thread_pool.join()

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_set)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_set))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        "returns one element from the data_set"
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = [self.bags[k] for k in indexes]

        f = [self.neighbors[k] for k in indexes]

        y = [self.bag_label[k] for k in indexes]

        return (X[0], f[0]), np.asarray(y, np.float32)

    def __data_generation(self, batch_train):
        """

        Parameters
        ----------
        batch_train:  a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches

        Returns
        -------
        bag_batch: a list of np.ndarrays of size (numnber of patches,h,w,d) , each of which contains the patches of an image
        neighbors: a list  of the adjacency matrices of size (numnber of patches,number of patches) of every image
        bag_label: an np.ndarray of size (number of patches,1) reffering to the label of the image

        """
        aug_batch = []
        img_data = batch_train[0]

        for i in range(img_data.shape[0]):
            ori_img = img_data[i, :, :, :]

            if self.shuffle:
                img = random_flip_img(ori_img, horizontal_chance=0.5, vertical_chance=0.5)
                img = random_rotate_img(img)

            else:
                img = ori_img

            exp_img = np.expand_dims(img, 0)

            aug_batch.append(exp_img)

        input_batch = np.concatenate(aug_batch)

        Idx = self._get_indices(batch_train[2], neighbors=self.k)
        if self.mode == "vaegan":

            values = self.generate_siamese_pairs(batch_train,Idx)

            adjacency_matrix = self.get_siamese_affinity(Idx, values)

        else:
            adjacency_matrix = self.get_knn_affinity(Idx)

        return input_batch, adjacency_matrix, batch_train[1]

    @tf.function
    def serve(self,x):
        return self.trained_model(x, training=False)

    def generate_siamese_pairs(self, batch,Idx):
        """

        Parameters
        ----------
        images :  np.ndarray of size (numnber of patches,h,w,d) contatining the pathes of an image
        Idx    : indices of the closest neighbors of every image

        Returns
        -------
        a list of np.ndarrays, pairing every patch of an image with its closest neighbors
        """

        values = []
        columns = (np.concatenate(np.asarray(Idx)).ravel())
        images=batch[0]
        label=batch[1]
        filenames=batch[2]

        rows = [[enum] * len(item) if isinstance(item, np.ndarray) else np.asarray([enum]) for enum, item in
                enumerate(Idx)]
        rows = np.concatenate(np.asarray(rows)).ravel()

        for row, column in zip(rows, columns):
            m1, s1=self.serve(np.expand_dims(images[int(row)], axis=0))
            m2, s2=self.serve(np.expand_dims(images[int(column)], axis=0))

            value=self.kl_mvn(m1.numpy().reshape(-1,1),np.exp(s1.numpy().reshape(-1,1)),m2.numpy().reshape(-1,1),np.exp(s2.numpy().reshape(-1,1)))

            values.append(value)

        values = [float(i) / max(values) for i in values]
        values = [1-x for x in values]


        return values

    def get_knn_affinity(self, Idx):
        """
        Create the adjacency matrix of each bag based on the euclidean distances between the patches
        Parameters
        ----------
        Idx:   a list of indices of the closest neighbors of every image

        Returns
        -------
        affinity:  an nxn np.ndarray that contains the neighborhood information for every patch.
        """

        affinity = np.zeros((Idx.shape[0], Idx.shape[0]), float)

        rows = np.asarray([[enum] * len(item) for enum, item in enumerate(Idx)]).ravel()

        columns = Idx.ravel()

        affinity[rows, columns] = 1

        return affinity

    def kl_mvn(self,m1, s1, m2, s2):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.


        From wikipedia
        KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                      (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        # store inv diag covariance of S1 and diff between means
        N = m1.shape[0]
        iS1 = 1 / s2
        diff = m2 - m1

        # kl is made of three terms
        tr_term = np.sum((1 / s2) * s1)
        det_term = np.log(np.prod(s2) / np.prod(s1))
        quad_term = np.sum((diff * diff) * iS1)

        return .5 * (tr_term + det_term + quad_term - N)


    def get_siamese_affinity(self, Idx, values):
        """
        Create   :    the adjacency matrix of each bag based on the distance scores produced by the siamese network

        Parameters
        ----------
        Idx       :  nxk np.ndarray containing the indices of the closest spatial neigbors of every patch
        train_set :  list of (patches, label, filenames) describing a bag

        Returns
        -------
        affinity : an nxn np.ndarray that contains the neighborhood information for every patch.

        """

        affinity = np.zeros((Idx.shape[0], Idx.shape[0]), float)

        columns = (np.concatenate(np.asarray(Idx)).ravel())

        rows = [[enum] * len(item) if isinstance(item, np.ndarray) else np.asarray([enum]) for enum, item in
                enumerate(Idx)]
        rows = np.concatenate(np.asarray(rows)).ravel().astype(int)

        affinity[rows, columns] = values

        np.fill_diagonal(affinity, 1)

        affinity=np.where(affinity < np.mean(values), 0, 1)

        affinity = affinity.astype("float32")

        return affinity

    def _get_indices(self, filenames, neighbors):
        """
        Computes the indices that correspond to the closest neighbors of each patch in a bag
        Parameters
        ----------
        filenames: list of filenames of all the patches in a bag. Each filename has the following format imgx-xposd-yposd-class,
        which enables the extraction of its spatial coordinates (xpos,ypos)
        neighbors: number of neigbors to be taken into account

        Returns
        -------
        neigbor_indices: nxk np.ndarray containing the indices of the closest spatial neigbors of every patch
        """

        coordinates = []

        for enum, each_path in enumerate(filenames):
            coords = get_coordinates(each_path)

            coordinates.append(coords)

        patch_distances = euclidean_distances(coordinates)

        neighbor_indices = np.argsort(patch_distances, axis=1)[:, :neighbors + 1]

        return np.asarray(neighbor_indices)

