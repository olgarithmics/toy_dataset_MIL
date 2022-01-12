import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer, multiply,LayerNormalization, Add, Dense, Dropout
import training.pooling_method as pooling

class Graph_Attention(Layer):
    """
    Implementation of the Graph Attention Mechanism

    # Arguments

        L_dim:              dimensionality of the attn_kernel_self matrix
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_gated:          boolean, whether use the gated attenion mechanism or not



    # Input Shape
        2D tensor with shape: (n, input_dim) corresponding to the feature representations h_1,h_2,....,h_n of every bag

    # Output Shape
        2D tensor with shape: (n, n) containing the relevance score between all the instances of a bag either connected or not

    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, use_gated=False,
                 **kwargs):

        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_gated = use_gated


        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(kernel_initializer)
        self.neighbor_weight_initializer = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(kernel_regularizer)
        self.neighbor_weight_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Graph_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        self.attn_kernel_self = self.add_weight(shape=(self.L_dim, 1),
                                                initializer=self.attn_kernel_initializer,
                                                name='attn_kernel_self',
                                                regularizer=self.attn_kernel_regularizer,
                                                trainable=True)

        self.attn_kernel_neighs = self.add_weight(shape=(self.L_dim, 1),
                                                  initializer=self.neighbor_weight_initializer,
                                                  name='attn_kernel_neigh',
                                                  regularizer=self.neighbor_weight_regularizer,
                                                  trainable=True)

        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True



    def call(self, input_tensor, mask=None):

        X=input_tensor

        features = K.tanh(K.dot(X, self.kernel))
        if self.use_gated:
            gate_x=K.sigmoid(K.dot(X,self.U))
            ac_x=features*gate_x
        else:
            ac_x=features

        attn_self = K.dot(ac_x, self.attn_kernel_self)

        attn_for_neighs = K.dot(ac_x, self.attn_kernel_neighs)

        data_input = attn_self + K.transpose(attn_for_neighs)

        data_input = K.tanh(data_input)


        return data_input

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.self.kernel_initializer),
            'attn_kernel_self': initializers.serialize(self.attn_kernel_self),
            'u_init': initializers.serialize(self.u_init),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'u_regularizer': regularizers.serialize(self.u_regularizer),
            'use_bias': self.use_bias,
            "use_gated": self.use_gated
        }
        base_config = super(Graph_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NeighborAggregator(Layer):
    """
    Aggregation of neighborhood information

    This layer is responsible for aggregatting the neighborhood information of the attentin matrix through the
    element-wise multiplication with an adjacency matrix. Every row of the produced
    matrix is averaged to produce a single attention score.

    # Arguments
        output_dim:            positive integer, dimensionality of the output space

    # Input shape
        2D tensor with shape: (n, n)
        2d tensor with shape: (None, None) correspoding to the adjacency matrix
    # Output shape
        2D tensor with shape: (1, units) corresponding to the attention coefficients of every instance in the bag
    """

    def __init__(self, output_dim,**kwargs):


        self.output_dim = output_dim


        super(NeighborAggregator, self).__init__(**kwargs)


    def sparse_mean(self,sparse_tensor,non_zero_elements):
        reduced_sum = tf.sparse.reduce_sum(sparse_tensor, 1)

        reduced_mean = tf.math.divide(
            reduced_sum, non_zero_elements, name=None)
        return reduced_mean


    def call(self, input_tensor,mask=None):
        data_input=input_tensor[0]

        adj_matrix=input_tensor[1]

        data_input = multiply([adj_matrix, data_input])

        non_zero_elements = tf.cast(tf.math.count_nonzero(adj_matrix, 1), tf.float32)

        sparse = tf.sparse.from_dense(data_input)

        sparse_mean = self.sparse_mean(sparse,non_zero_elements)

        x=tf.reshape(tensor=sparse_mean, shape=(tf.shape(data_input)[1],))

        alpha = K.softmax(x)

        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class Last_Sigmoid(Layer):
    """
    Attention Activation

    This layer contains the last sigmoid layer of the network


    # Arguments
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_bias:           boolean, whether use bias or not

    # Input shape
        2D tensor with shape: (n, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 pooling_mode="sum",
                 kernel_regularizer=None, bias_regularizer=None,
                 use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.pooling_mode=pooling_mode
        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):

        x  = pooling.choice_pooling(x, self.pooling_mode)
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)
        out = K.sigmoid(x)
        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistanceLayer(Layer):

    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim

        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        x, y = input_tensor

        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(DistanceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Score_pooling(Layer):
    """
    Score pooling layer

    This layer contains a FC layer which only has one neural with sigmoid actiavtion
    and MIL pooling. The input of this layer is instance features. Then we obtain
    instance scores via this FC layer. And use MIL pooling to aggregate instance scores
    into bag score that is the output of Score pooling layer.
    This layer is used in mi-Net.

    # Arguments
        output_dim: Positive integer, dimensionality of the output space
        kernel_initializer: Initializer of the `kernel` weights matrix
        bias_initializer: Initializer of the `bias` weights
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the `bias` weights
        use_bias: Boolean, whether use bias or not
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None,
                    use_bias=True, pooling_mode='max', **kwargs):
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Score_pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape

        # compute instance-level score
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)

        # sigmoid
        x = K.sigmoid(x)

        # do-pooling operator
        output = pooling.choice_pooling(x, self.pooling_mode)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias,
            'pooling_mode': self.pooling_mode
        }
        base_config = super(Score_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class RC_block(Layer):
    """
    Residual Connection block

    This layer contains a MIL pooling with the layer input to produce a tensor of
    outputs (bag representation residuals).
    This layer is used in MI-Net with RC.

    # Arguments
        pooling_mode: A string,
                      the mode of MIL pooling method, like 'max' (max pooling),
                      'ave' (average pooling), 'lse' (log-sum-exp pooling)

    # Input shape
        2D tensor with shape: (batch_size, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """
    def __init__(self, pooling_mode='max', **kwargs):
        self.pooling_mode = pooling_mode
        super(RC_block, self).__init__(**kwargs)

    def call(self, x, mask=None):
        n, d = x.shape

        # do-pooling operator

        output = pooling.choice_pooling(x, self.pooling_mode)

        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape)

    def get_config(self):
        config = {'pooling_mode':self.pooling_mode}
        base_config = super(RC_block, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Feature_pooling(Layer):
        """
        Feature pooling layer

        This layer contains a MIL pooling and a FC layer which only has one neural with
        sigmoid activation. The input of this layer is instance features. Via MIL pooling,
        we aggregate instance features to bag features. Finally, we obtain bag score by
        this FC layer with only one neural and sigmoid activation
        This layer is used in MI-Net and MI-Net with DS.

        # Arguments
            output_dim: Positive integer, dimensionality of the output space
            kernel_initializer: Initializer of the `kernel` weights matrix
            bias_initializer: Initializer of the `bias` weights
            kernel_regularizer: Regularizer function applied to the `kernel` weights matrix
            bias_regularizer: Regularizer function applied to the `bias` weights
            use_bias: Boolean, whether use bias or not
            pooling_mode: A string,
                          the mode of MIL pooling method, like 'max' (max pooling),
                          'ave' (average pooling), 'lse' (log-sum-exp pooling)

        # Input shape
            2D tensor with shape: (batch_size, input_dim)
        # Output shape
            2D tensor with shape: (1, units)
        """

        def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None,
                     use_bias=True, pooling_mode='max', **kwargs):
            self.output_dim = output_dim
            self.pooling_mode = pooling_mode

            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)

            self.use_bias = use_bias
            super(Feature_pooling, self).__init__(**kwargs)

        def build(self, input_shape):
            assert len(input_shape) == 2
            input_dim = input_shape[1]

            self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.output_dim,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            trainable=True)
            else:
                self.bias = None

            self.input_built = True

        def call(self, x, mask=None):
            n, d = x.shape

            # do-pooling operator
            x = pooling.choice_pooling(x, self.pooling_mode)

            # compute bag-level score
            output = K.dot(x, self.kernel)
            if self.use_bias:
                output = K.bias_add(output, self.bias)
            output = K.sigmoid(output)

            return output

        def compute_output_shape(self, input_shape):
            shape = list(input_shape)
            assert len(shape) == 2
            shape[1] = self.output_dim
            return tuple(shape)

        def get_config(self):
            config = {
                'output_dim': self.output_dim,
                'kernel_initializer': initializers.serialize(self.kernel.initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'use_bias': self.use_bias,
                'pooling_mode': self.pooling_mode
            }
            base_config = super(Feature_pooling, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

class DP_pooling(Layer):
    def __init__(self,output_dim,T, **kwargs):

        self.output_dim = output_dim
        self.T=T

        super(DP_pooling, self).__init__(**kwargs)

    def call(self, input_tensor,mask=None):
        X = input_tensor

        @tf.function
        def update_dyn_pool(T,X):

            def squashing(v):
               u = tf.pow(v, 2)
               sum_ = tf.reduce_sum(u, axis = 1, keepdims = True)

               left_ = sum_ / (sum_ + 1.0) #64*1
               right_ = tf.nn.l2_normalize(v, axis = 1) #64*200
               out = left_ * right_
               return out

            initial_state = tf.zeros(tf.shape(X)[0], dtype=tf.dtypes.float32)
            state=initial_state

            max_seq_len = tf.shape(X)[0]
            states = tf.TensorArray(tf.float32, size=max_seq_len, dynamic_size=True, infer_shape=True)

            for i in tf.range(T):
                c_t = K.softmax(K.transpose(state))

                s_t = multiply([X, c_t], name='dyn_mil')
                squash_t = squashing(s_t)

                state = state + tf.reduce_sum(tf.multiply(X, squash_t), axis=1)
                states=states.write(i,squash_t)

            return states.read(T-1)

        squash= update_dyn_pool(self.T,X)

        out =K.sqrt(K.maximum(tf.reduce_sum(K.square(squash),keepdims=True), K.epsilon()))

        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            "output_dim" :self.output_dim,
            "T":self.T
        }
        base_config = super(DP_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        #self.wv = tf.keras.layers.Dense(d_model)
        #self.dense = tf.keras.layers.Dense(d_model)


    def scaled_dot_product_attention(self, q, k,mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk
        # add the mask to the scaled tensor.
        if mask is not None: scaled_attention_logits += (mask * -1e9)

        return  scaled_attention_logits

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, input_tensor, mask=None):

        self.q=input_tensor
        self.k=input_tensor
        q = self.wq(self.q)  # (batch_size, seq_len, d_model)
        k = self.wk(self.k)  # (batch_size, seq_len, d_model)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = self.scaled_dot_product_attention(q, k, mask)

        return  attention_weights







