import glob
import os
import re
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from tensorflow.keras.callbacks import  EarlyStopping
from tensorflow.python.keras.callbacks import CallbackList as KerasCallbackList
from tensorflow.keras.layers import Input,Flatten, Dense, Dropout,Add,Average,LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from dataloaders.DataGenerator import DataGenerator
from tensorflow.keras.optimizers import RMSprop
from training.custom_layers import NeighborAggregator, Graph_Attention, Last_Sigmoid, MultiHeadAttention,DistanceLayer,multiply, Score_pooling, Feature_pooling, RC_block, DP_pooling
from dataloaders.dataset import Get_train_valid_Path
from training.metrics import bag_accuracy, bag_loss
from training.stack_layers import stack_layers, make_layer_list
from dataloaders.BreastCancerDataset import BreastCancerDataset
from dataloaders.ColonCancerDataset import ColonCancerDataset
from vaegan_utils.models import create_models, build_graph
from tensorflow.keras.losses import BinaryCrossentropy
from vaegan_utils.training import fit_models
from vaegan_utils.dataloader import encoder_loader, decoder_loader, discriminator_loader, ImgIterator, load_images
from vaegan_utils.callbacks import DecoderSnapshot, ModelsCheckpoint
import tensorflow as tf
import time


class VaeGan:
    def __init__(self,args):
        """
        Build the architecture of the siamese net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """
        self.initial_epoch=args.initial_epoch
        self.vaegan_save_dir=args.vaegan_save_dir
        self.decode_dir=args.decode_dir
        self.encoder, self.decoder, self.discriminator = create_models()

        self.encoder_train, self.decoder_train, self.discriminator_train, self.vae, self.vaegan = build_graph(self.encoder, self.decoder, self.discriminator)


        self.epoch_format = '.{epoch:03d}.h5'

        os.makedirs(self.vaegan_save_dir, exist_ok=True)
        os.makedirs(self.decode_dir, exist_ok=True)


        if args.initial_epoch != 0:
            suffix = self.epoch_format.format(epoch=args.initial_epoch)
            self.encoder.load_weights('encoder' + suffix)
            self.decoder.load_weights('decoder' + suffix)
            self.discriminator.load_weights('discriminator' + suffix)


    def train(self, train_bags,val_bags, irun, ifold):
        """
        Train the siamese net

        Parameters
        ----------
        pairs_train : a list of lists, each of which contains an np.ndarray of the patches of each image,
        the label of each image and a list of filenames of the patches
        check_dir   : str, specifying the directory where weights of the siamese net are going to be stored
        irun        : int reffering to the id of the experiment
        ifold       : fold reffering to the fold of the k-cross fold validation

        Returns
        -------
        A History object containing a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values

        """


        def set_trainable(model, trainable):
            model.trainable = trainable
            for layer in model.layers:
                layer.trainable = trainable

        real_acc = tf.keras.metrics.BinaryAccuracy(name="real_acc")
        fake_zp_acc = tf.keras.metrics.BinaryAccuracy(name="fake_z_acc")
        fake_z_acc = tf.keras.metrics.BinaryAccuracy(name="fake_zp_acc")

        rmsprop = RMSprop(learning_rate=0.0003)
        set_trainable(self.encoder, False)
        set_trainable(self.decoder, False)
        self.discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, [real_acc, fake_z_acc, fake_zp_acc])
        self.discriminator_train.summary()

        decoder_z_acc = tf.keras.metrics.BinaryAccuracy(name="decoder_z_acc")
        decoder_zp_acc = tf.keras.metrics.BinaryAccuracy(name="decoder_zp_acc")

        set_trainable(self.discriminator, False)
        set_trainable(self.decoder, True)
        self.decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2, [decoder_z_acc, decoder_zp_acc])
        self.decoder_train.summary()

        set_trainable(self.decoder, False)
        set_trainable(self.encoder, True)
        self.encoder_train.compile(rmsprop)
        self.encoder_train.summary()

        set_trainable(self.vaegan, True)

        try:
            os.makedirs("{}/irun{}_ifold{}".format(self.vaegan_save_dir,irun, ifold), exist_ok=True)
            print("Directory '%s' created successfully" % "{}/irun{}_ifold{}".format(self.vaegan_save_dir,irun, ifold))
        except OSError as error:
            print("Directory '%s' can not be created")

        checkpoint = ModelsCheckpoint("{}/irun{}_ifold{}/".format(self.vaegan_save_dir,irun, ifold), self.epoch_format,self.encoder, self.decoder, self.discriminator)
        decoder_sampler = DecoderSnapshot("{}/".format(self.decode_dir))

        callbacks = [checkpoint, decoder_sampler]

        epochs = 70

        seed = np.random.randint(2 ** 32 - 1)

        batch_size = 128

        hdf5Iterator = ImgIterator(np.concatenate((train_bags, val_bags)), batch_size=batch_size, shuffle=True)
        steps_per_epoch = len(hdf5Iterator)
        img_loader = load_images(hdf5Iterator, num_child=3)

        dis_loader = discriminator_loader(img_loader, seed=seed)
        dec_loader = decoder_loader(img_loader, seed=seed)
        enc_loader = encoder_loader(img_loader)

        models = [self.discriminator_train, self.decoder_train, self.encoder_train]

        generators = [dis_loader, dec_loader, enc_loader]

        fit_models(self.vaegan, models, generators, batch_size=batch_size,
                               steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                               epochs=epochs, initial_epoch=self.initial_epoch)
        return self.encoder



class GraphAttnet:
    def __init__(self, args, useMulGpue=False):
        """
        Build the architercure of the Graph Att net
        Parameters
        ----------
        arch:          a dict describing the architecture of the neural network to be trained
        mode            :str, specifying the version of the model (siamese, euclidean)
        containing input_types and input_placeholders for each key and value pair, respecively.
        input_shape:   tuple(int, int, int) of the input shape of the patches
        useMulGpue:    boolean, whether to use multi-gpu processing or not
        """

        self.args=args
        self.arch=args.arch
        self.mode=args.mode
        self.input_shape = tuple(args.input_shape)
        self.data=args.data
        self.weight_file=args.weight_file
        self.k=args.k
        self.save_dir=args.save_dir
        self.experiment_name=args.experiment_name
        self.weight_decay=args.weight_decay
        self.pooling_mode=args.pooling_mode
        self.init_lr=args.init_lr
        self.epochs=args.epochs


        self.inputs = {
            'bag': Input(self.input_shape),
            'adjacency_matrix': Input(shape=(None,), dtype='float32', name='adjacency_matrix'),
        }

        self.useMulGpu = useMulGpue
        self.layers = []
        self.layers += make_layer_list(self.arch, 'graph', self.weight_decay)

        self.outputs = stack_layers(self.inputs, self.layers)

        # neigh = Graph_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(self.weight_decay),
        #                       name='neigh',
        #                       use_gated=args.useGated)(self.outputs["bag"])
        neigh = MultiHeadAttention(d_model=256, num_heads=1)(self.outputs["bag"])

        alpha = NeighborAggregator(output_dim=1, name="alpha")([neigh, self.inputs["adjacency_matrix"]])

        x_mul = multiply([alpha, self.outputs["bag"]], name="mul")

        out = Last_Sigmoid(output_dim=1, name='FC1_sigmoid',pooling_mode=self.pooling_mode)(x_mul)

        self.net = Model(inputs=[self.inputs["bag"], self.inputs["adjacency_matrix"]], outputs=[out])

        self.net.compile(optimizer=Adam(lr=self.init_lr, beta_1=0.9, beta_2=0.999), loss=bag_loss,
                            metrics=[bag_accuracy])

    @property
    def model(self):
        return self.net

    def load_siamese(self, irun, ifold):
        """
        Loads the appropriate siamese model using the information of the fold of k-cross
        fold validation and the id of experiment
        Parameters
        ----------
        check_dir  : directory where the weights of the pretrained siamese network. Weight files are stored in the format:
        weights-irun:d-ifold:d.hdf5
        irun       : int referring to the id of the experiment
        ifold      : int referring to the fold from the k-cross fold validation

        Returns
        -------
        returns  a Keras model instance of the pre-trained siamese net
        """
        encoder, decoder, discriminator = create_models()

        def extract_number(f):
            s = re.findall("\d+\.\d+", f)
            return ((s[0]) if s else -1, f)

        file_paths = glob.glob(os.path.join( "vaegan_weights/irun{}_ifold{}".format(irun, ifold), 'discriminator*'))
        file_paths.reverse()
        file_path = (max(file_paths, key=extract_number))


        discriminator.load_weights(file_path)
        return discriminator

    def train(self,train_bags , irun, ifold,detection_model):
        """
        Train the Graph Att net
        Parameters
        ----------
        train_set       : a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches
        check_dir       :str, specifying directory where the weights of the siamese net are stored
        irun            :int, id of the experiment
        ifold           :int, fold of the k-corss fold validation
        weight_file     :boolen, specifying whether there is a weightflie or not

        Returns
        -------
        A History object containing  a record of training loss values and metrics values at successive epochs,
        as well as validation loss values and validation metrics values.
        """


        train_bags, val_bags = Get_train_valid_Path(train_bags, ifold, train_percentage=0.9)
        if self.data=='colon':
            model_val_set = ColonCancerDataset(patch_size=27,augmentation=False).load_bags(val_bags)
            model_train_set = ColonCancerDataset(patch_size=27,augmentation=True).load_bags(train_bags)
        else :

            model_val_set=BreastCancerDataset(format='.tif', patch_size=128,
                                stride=16, augmentation=False, model=detection_model).load_bags(wsi_paths=val_bags)
            model_train_set = BreastCancerDataset(format='.tif', patch_size=128,
                                               stride=16, augmentation=True, model=detection_model).load_bags(wsi_paths=train_bags)

        if self.mode=="vaegan":
            if self.weight_file:
                self.discriminator_test = self.load_siamese( irun, ifold)
            else:

                self.vaegan_net_test= VaeGan(self.args)
                self.discriminator_test=self.vaegan_net_test.train(model_train_set,model_val_set, irun=irun, ifold=ifold)


            train_gen = DataGenerator(batch_size=1, data_set=model_train_set, k=self.k, shuffle=True, mode=self.mode,
                                      trained_model=self.discriminator_test)

            val_gen = DataGenerator(batch_size=1, data_set=model_val_set, k=self.k, shuffle=False, mode=self.mode,
                                    trained_model=self.discriminator_test)

        else:
            train_gen = DataGenerator(batch_size=1, data_set=model_train_set, k=self.k, shuffle=True, mode=self.mode)

            val_gen = DataGenerator(batch_size=1, data_set=model_val_set, k=self.k, shuffle=False, mode=self.mode)

        os.makedirs(self.save_dir, exist_ok=True)

        checkpoint_path = os.path.join(self.save_dir, "{}.ckpt".format(self.experiment_name))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         monitor='val_loss',
                                                         save_weights_only=True,
                                                         save_best_only=True,
                                                         mode='auto',
                                                         save_freq='epoch',
                                                         verbose=1)

        _callbacks = [EarlyStopping(monitor='val_loss', patience=20), cp_callback]
        callbacks = KerasCallbackList(_callbacks, add_history=True, model=self.net)

        logs = {}
        callbacks.on_train_begin(logs=logs)

        optimizer = Adam(learning_rate=self.init_lr, beta_1=0.9, beta_2=0.999)
        loss_fn = BinaryCrossentropy(from_logits=False)
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_acc_metric = tf.keras.metrics.BinaryAccuracy()

        @tf.function(experimental_relax_shapes=True)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = self.net(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.net.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
            train_acc_metric.update_state(y, logits)
            return loss_value

        @tf.function(experimental_relax_shapes=True)
        def val_step(x, y):
            val_logits = self.net(x, training=False)
            val_loss = loss_fn(y, val_logits)
            val_acc_metric.update_state(y, val_logits)
            return val_loss

        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_gen):

                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_train_batch_begin(step, logs=logs)
                loss_value = train_step(x_batch_train, np.expand_dims(y_batch_train, axis=0))

                logs["train_loss"] = loss_value

                callbacks.on_train_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)
                if step % 20 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            train_acc_metric.reset_states()

            for step, (x_batch_val, y_batch_val) in enumerate(val_gen):
                callbacks.on_batch_begin(step, logs=logs)
                callbacks.on_test_batch_begin(step, logs=logs)
                vall_loss = val_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))
                logs["val_loss"] = vall_loss

                callbacks.on_test_batch_end(step, logs=logs)
                callbacks.on_batch_end(step, logs=logs)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            callbacks.on_epoch_end(epoch, logs=logs)

        callbacks.on_train_end(logs=logs)

        history_object = None
        for cb in callbacks:
            if isinstance(cb, tf.keras.callbacks.History):
                history_object = cb
        assert history_object is not None


        return history_object

    def predict(self,test_bags, detection_model, test_model, irun, ifold):

        """
        Evaluate the test set
        Parameters
        ----------
        test_set: a list of lists, each of which contains the np.ndarray of the patches of each image,
        the label of the image and a list of filenames of the patches

        Returns
        -------

        test_loss : float reffering to the test loss
        acc       : float reffering to the test accuracy
        precision : float reffering to the test precision
        recall    : float referring to the test recall
        auc       : float reffering to the test auc


        """

        if self.data=="colon":
            test_set = ColonCancerDataset(patch_size=27,augmentation=False).load_bags(wsi_paths=test_bags)
        else:
            test_set = BreastCancerDataset(format='.tif', patch_size=128,
                                                  stride=16, augmentation=False, model=detection_model).load_bags(wsi_paths=test_bags)

        if self.mode=="vaegan":
            self.discriminator_test = self.load_siamese(irun, ifold)
            test_gen = DataGenerator(batch_size=1, data_set=test_set, k=self.k, shuffle=False, mode=self.mode,
                                 trained_model=self.discriminator_test)
        else:
            test_gen = DataGenerator(batch_size=1, data_set=test_set, k=self.k, shuffle=False, mode=self.mode)

        loss_value=[]
        test_loss_fn = BinaryCrossentropy(from_logits=False)
        eval_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        checkpoint_path = os.path.join(self.save_dir, "{}.ckpt".format(self.experiment_name))
        test_model.load_weights(checkpoint_path)

        @tf.function(experimental_relax_shapes=True)
        def test_step(images, labels):

            predictions = test_model(images, training=False)
            test_loss = test_loss_fn(labels, predictions)

            eval_accuracy_metric.update_state(labels, predictions)
            return test_loss,predictions

        y_pred = []
        y_true = []
        for x_batch_val, y_batch_val in test_gen:

            test_loss, pred = test_step(x_batch_val, np.expand_dims(y_batch_val, axis=0))
            loss_value.append(test_loss)
            y_true.append(y_batch_val)
            y_pred.append(pred.numpy().tolist()[0][0])

        test_loss = np.mean(loss_value)
        print("Test loss: %.4f" % (float(test_loss),))

        test_acc = eval_accuracy_metric.result()
        print("Test acc: %.4f" % (float(test_acc),))

        auc = roc_auc_score(y_true, y_pred)
        print("AUC {}".format(auc))

        precision = precision_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("precision {}".format(precision))

        recall = recall_score(y_true, np.round(np.clip(y_pred, 0, 1)))
        print("recall {}".format(recall))

        return test_loss,test_acc, auc, precision, recall

    def visualize_conv_layer(self,layer_name, data_name, test_img, detection_model, irun, ifold,saved_weights_dir=None):


        if data_name == "colon":
            test_set = ColonCancerDataset(patch_size=27, augmentation=False).load_bags(wsi_paths=[test_img])
        else:
            test_set = BreastCancerDataset(format='.tif', patch_size=128,
                                           stride=16, augmentation=False, model=detection_model).load_bags(wsi_paths=test_img)

        if self.mode == "vagean":

            self.discriminator_test = self.load_siamese( irun, ifold)
            test_gen = DataGenerator(batch_size=1, data_set=test_set, k=self.k, shuffle=False, mode=self.mode,
                                     trained_model=self.discriminator_test )
        else:
            test_gen = DataGenerator(batch_size=1, data_set=test_set, k=self.k, shuffle=False, mode=self.mode)

        layer_output = self.net.get_layer(layer_name).output

        intermediate_model = Model(inputs=self.net.input, outputs=layer_output)
        intermediate_model.load_weights(saved_weights_dir, by_name=True)

        intermediate_prediction = intermediate_model.predict_on_batch(test_gen[0][0])

        return intermediate_prediction