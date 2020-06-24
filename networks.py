"""
Implementation of network architectures
"""
from __future__ import absolute_import
from __future__ import print_function

import tensorflow.compat.v1.keras.backend as K
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Input
from tensorflow.compat.v1.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.compat.v1.keras.layers import GRU, CuDNNGRU, Bidirectional
from tensorflow.compat.v1.keras.layers import Reshape, RepeatVector
from tensorflow.compat.v1.keras.layers import AveragePooling2D, Lambda, Flatten, Dense
from tensorflow.compat.v1.keras.layers import ZeroPadding2D, MaxPooling2D
from tensorflow.compat.v1.keras.layers import Activation, Dropout
## add and Concatenate now also live in the 'layers' module, not in layers.merge
from tensorflow.compat.v1.keras.layers import add, Concatenate
## ============================================================================
from tensorflow.compat.v1.keras.regularizers import l2
from tensorflow.compat.v1.keras.initializers import Constant, TruncatedNormal


def allcnn(image_shape, n_classes, dropout=None, weight_decay=None,
           batch_norm=False, depth='orig', id_output=False,
           input_dropout=False):
    """
    Defines the All convolutional network (All-CNN), originally described in
    https://arxiv.org/abs/1412.6806. It is implemented in a modular way in
    order to allow for more flexibility, for instance modifying the depth of
    the network.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : float
        If not None, the rate of Dropout regularization

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    depth : str
        Allows to specify a shallower or deeper architecture:
            - 'shallower': 1 convolutional block (A) + 1 output block (B)
            - 'deeper': 2 convolutional blocks (A) + 1 output block (B)

    id_output : bool
        If True, a matrix of squared pairwise distances will be computed with
        the activations of each convoltional layer (after ReLU) and appended as
        output of the model.

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, mse_list, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (width) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   input_shape=image_shape,
                   name='conv%d' % n_layer)(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv%dbn' % n_layer)(x)

        # ReLU Activation
        x = Activation('relu', name='conv%drelu' % n_layer)(x)    

        # MSE layer: matrix of squared pairwise distances of the activations
        if id_output:
            x_flatten = Flatten()(x)
            mse = Lambda(pairwise_mse, 
                         name='conv{}mse'.format(n_layer))(x_flatten)
            daug_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                              name='daug_inv{}'.format(n_layer))(mse)
            class_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                               name='class_inv{}'.format(n_layer))(mse)
            mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                               name='mean_inv{}'.format(n_layer))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        return x, mse_list

    def block_a(x, mse, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 3x3 conv with stride 1
            3. 3x3 conv with stride 2
        """
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 2)
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=2, 
                            padding='same', n_layer=(n_block - 1) * 3 + 3)

        return x, mse

    def block_b(x, mse, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 1x1 conv with stride 1 and valid padding
            3. 1x1 conv with stride 1 and valid padding
        """
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x, mse = conv_layer(x, mse, filters, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 2)
        x, mse = conv_layer(x, mse, n_classes, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 3)

        return x, mse

    inputs = Input(shape=image_shape)
    mse_list = []

    # Dropout of 20 %
    if dropout is not None and input_dropout == True:
        x = Dropout(rate=0.2, name='dropout0')(inputs)

        # Block 1: 96 filters
        x, mse_list = block_a(x, mse_list, filters=96, n_block=1)
    else:
        # Block 1: 96 filters
        x, mse_list = block_a(inputs, mse_list, filters=96, n_block=1)

    # Dropout of 50 %
    if dropout:
        x = Dropout(rate=dropout, name='dropout1')(x)

    # Block 2: 192 filters
    if depth == 'shallower':
        # Pre-logits block
        x, mse_list = block_b(x, mse_list, filters=192, n_block=2)
    else:
        x, mse_list = block_a(x, mse_list, filters=192, n_block=2)

        # Dropout of 50 %
        if dropout:
            x = Dropout(rate=dropout, name='dropout2')(x)

        # Block 3: 192 filters
        if depth == 'deeper':
            x, mse_list = block_a(x, mse_list, filters=192, n_block=3)
            # Pre-logits block
            x, mse_list = block_b(x, mse_list, filters=192, n_block=4)
        else:
            # Pre-logits block
            x, mse_list = block_b(x, mse_list, filters=192, n_block=3)

    # Global Average Pooling
    logits = GlobalAveragePooling2D(name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[predictions] + mse_list)
    else:
        model = Model(inputs=inputs, outputs=predictions)

    return model


def allcnn_large(image_shape, n_classes, dropout=None, weight_decay=None,
                 batch_norm=False, depth='orig', stride_conv1=4, 
                 id_output=False, input_dropout=False):
    """
    Defines the ImageNet version of the All convolutional network (All-CNN),
    originally described in https://arxiv.org/abs/1412.6806. It is implemented
    in a modular way in order to allow for more flexibility, for instance
    modifying the depth of the network.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : float
        If not None, the rate of Dropout regularization

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    depth : str
        Allows to specify a shallower or deeper architecture:
            - 'shallower': 1 convolutional block (A) + 1 output block (B)
            - 'deeper': 2 convolutional blocks (A) + 1 output block (B)

    id_output : bool
        If True, a matrix of squared pairwise distances will be computed with
        the activations of each convoltional layer (after ReLU) and appended as
        output of the model.

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, mse_list, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (width) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   input_shape=image_shape,
                   name='conv%d' % n_layer)(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv%dbn' % n_layer)(x)

        # ReLU Activation
        x = Activation('relu', name='conv%drelu' % n_layer)(x)    

        # MSE layer: matrix of squared pairwise distances of the activations
        if id_output:
            x_flatten = Flatten()(x)
            mse = Lambda(pairwise_mse, 
                         name='conv{}mse'.format(n_layer))(x_flatten)
            daug_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                              name='daug_inv{}'.format(n_layer))(mse)
            class_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                               name='class_inv{}'.format(n_layer))(mse)
            mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                               name='mean_inv{}'.format(n_layer))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        return x, mse_list

    def block_a(x, mse, k1, s1, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. k1xk1 conv with stride s1
            2. 1x1 conv with stride 1
            3. 3x3 conv with stride 2
        """
        x, mse = conv_layer(x, mse, filters, kernel=k1, stride=s1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x, mse = conv_layer(x, mse, filters, kernel=1, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 2)
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=2, 
                            padding='same', n_layer=(n_block - 1) * 3 + 3)

        return x, mse

    def block_b(x, mse, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 1x1 conv with stride 1 and valid padding
            3. 1x1 conv with stride 1 and valid padding
        """
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x, mse = conv_layer(x, mse, filters, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 2)
        x, mse = conv_layer(x, mse, n_classes, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 3)

        return x, mse

    inputs = Input(shape=image_shape)
    mse_list = []

    # Dropout of 20 %
    if dropout is not None and input_dropout == True:
        x = Dropout(rate=0.2, name='dropout0')(inputs)

        # Block 1: 96 filters
        x, mse_list = block_a(x, mse_list, filters=96, k1=11, s1=stride_conv1, 
                              n_block=1)
    else:
        # Block 1: 96 filters
        x, mse_list = block_a(inputs, mse_list, filters=96, k1=11, 
                              s1=stride_conv1, n_block=1)

    # Block 2: 256 filters
    x, mse_list = block_a(x, mse_list, filters=256, k1=5, s1=1, n_block=2)

    # Block 3: 384 filters
    x, mse_list = block_a(x, mse_list, filters=384, k1=3, s1=1, n_block=3)

    # Dropout of 50 %
    if dropout:
        x = Dropout(rate=dropout, name='dropout1')(x)

    # Block 4: 1024 filters
    x, mse_list = block_b(x, mse_list, filters=1024, n_block=4)

    # Global Average Pooling
    logits = GlobalAveragePooling2D(name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[predictions] + mse_list)
    else:
        model = Model(inputs=inputs, outputs=predictions)

    return model


def allcnn_mnist(image_shape, n_classes, dropout=False, weight_decay=None,
                 batch_norm=False, depth='orig', id_output=False,
                 input_dropout=False):
    """
    Defines the All convolutional network (All-CNN), originally described in
    https://arxiv.org/abs/1412.6806. It is implemented in a modular way in
    order to allow for more flexibility, for instance modifying the depth of
    the network.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    depth : str
        Allows to specify a shallower or deeper architecture:
            - 'shallower': 1 convolutional block (A) + 1 output block (B)
            - 'deeper': 2 convolutional blocks (A) + 1 output block (B)

    id_output : bool
        If True, a matrix of squared pairwise distances will be computed with
        the activations of each convoltional layer (after ReLU) and appended as
        output of the model.

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, mse_list, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (width) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   input_shape=image_shape,
                   name='conv%d' % n_layer)(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv%dbn' % n_layer)(x)

        # ReLU Activation
        x = Activation('relu', name='conv%drelu' % n_layer)(x)    

        # MSE layer: matrix of squared pairwise distances of the activations
        if id_output:
            x_flatten = Flatten()(x)
            mse = Lambda(pairwise_mse, name='conv%dmse' % n_layer)(x_flatten)
            mse_list.append(mse)

        return x, mse_list

    def block_a(x, mse, filters, n_block):
        """
        Defines a block of 2 convolutional layers:
            1. 3x3 conv with stride 1
            2. 3x3 conv with stride 2
        """
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 2 + 1)
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=2, 
                            padding='same', n_layer=(n_block - 1) * 2 + 2)

        return x, mse

    def block_b(x, mse, filters, n_block):
        """
        Defines a block of 2 convolutional layers:
            1. 3x3 conv with stride 1
            2. 1x1 conv with stride 1 and valid padding
        """
        x, mse = conv_layer(x, mse, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 2 + 1)
        x, mse = conv_layer(x, mse, n_classes, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 2 + 2)

        return x, mse

    inputs = Input(shape=image_shape)
    mse_list = []

    # Dropout of 20 %
    if dropout & input_dropout:
        x = Dropout(rate=0.2, name='dropout0')(inputs)

        # Block 1: 96 filters
        x, mse_list = block_a(x, mse_list, filters=96, n_block=1)
    else:
        # Block 1: 96 filters
        x, mse_list = block_a(inputs, mse_list, filters=96, n_block=1)

    # Dropout of 50 %
    if dropout:
        x = Dropout(rate=0.5, name='dropout1')(x)

    # Block 2: 192 filters
    if depth == 'shallower':
        # Pre-logits block
        x, mse_list = block_b(x, mse_list, filters=192, n_block=2)
    else:
        x, mse_list = block_a(x, mse_list, filters=192, n_block=2)

        # Dropout of 50 %
        if dropout:
            x = Dropout(rate=0.5, name='dropout2')(x)

        # Block 3: 192 filters
        if depth == 'deeper':
            x, mse_list = block_a(x, mse_list, filters=192, n_block=3)
            # Pre-logits block
            x, mse_list = block_b(x, mse_list, filters=192, n_block=4)
        else:
            # Pre-logits block
            x, mse_list = block_b(x, mse_list, filters=192, n_block=3)

    # Global Average Pooling
    logits = GlobalAveragePooling2D(name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[predictions] + mse_list)
    else:
        model = Model(inputs=inputs, outputs=predictions)

    return model


def wrn(image_shape, n_classes, dropout=False, weight_decay=None,
        batch_norm=False, blocks_per_group=4, widening_factor=10, 
        stride_conv1=1, id_output=False):
    """
    Defines the wide residual network (WRN), originally described in
    https://arxiv.org/abs/1605.07146. It is implemented in a modular way in
    order to allow for more flexibility.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    blocks_per_group : int
        Number of residual blocks per group. Each new group is defined to be
        wider than the previous one (16 * k, 32 * k, 64 * k).

    widening_factor : int
        Factor (k) by which the width of the residual blocks is multiplied. 

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def residual_block(x, mse_list, dropout=False, weight_decay=None, 
                       batch_norm=False, n_filters=16, subsample_factor=1, 
                       group=0, block=0):
        """
        Defines a residual block according to https://arxiv.org/abs/1605.07146:
              x --> BN - ReLU - 3x3 Conv - BN - ReLU - 3x3 Conv --> + --> out
              |                                                     ^
              |                                                     |  
              -------------------------------------------------------

        Parameters
        ----------
        x : Tensor
            Input to the block

        dropout : bool
            If True, Dropout regularization is included

        weight_decay : float
            L2 factor for the weight decay regularization or None

        batch_norm : bool
            If True, batch normalization is added before every ReLU activation

        n_filters : int 
            Number of filters for the convolutional layer

        subsample_factor : int
            Factor by which the input feature maps are downsampled. If larger 
            than 1, the shortcut connection is pooled by a 2D average 
            operation.

        group : int
            Group number. Used for layer naming.

        block : int
            Block number. Used for layer naming

        Returns
        -------
        out : Tensor
            Output of the residual block
        """

        def zero_pad_channels(x, pad=0):
            """
            Zero-pads an input to match the size of the next block.

            Parameters
            ----------
            x : Tensor
                Input to the block

            pad : int
                Amount of padding

            Returns
            -------
                Zero-padded input
            """
            import tensorflow as tf
            pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]

            return tf.pad(x, pattern)

        prev_n_channels = K.int_shape(x)[3]

        if subsample_factor > 1:
            subsample = (subsample_factor, subsample_factor)
            # Shortcut connection: subsample + zero-pad channel dim
            shortcut = AveragePooling2D(pool_size=subsample)(x)
        else:
            subsample = (1, 1)
            # Shortcut connection: identity
            shortcut = x

        if n_filters > prev_n_channels:
            shortcut = Lambda(zero_pad_channels,
                              arguments={'pad': n_filters - prev_n_channels})\
                              (shortcut)

        n_layer = ((group - 1) * blocks_per_group + (block - 1)) * 2 + 1
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='g%db%dconv%dbn' \
                                           % (group, block, n_layer))(x)
        x = Activation('relu', 
                       name='g%db%dconv%drelu' % (group, block, n_layer))(x)
        x = Conv2D(filters=n_filters,
                   kernel_size=(3, 3),
                   strides=subsample,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   name='g%db%dconv%d' % (group, block, n_layer))(x)
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='g%db%dconv%dbn' \
                                           % (group, block, n_layer+1))(x)
        x = Activation('relu', 
                       name='g%db%dconv%drelu' % (group, block, n_layer+1))(x)
        if dropout:
            x = Dropout(0.3)(x)
        x = Conv2D(filters=n_filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   use_bias=True,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   name='g%db%dconv%d' % (group, block, n_layer+1))(x)
        x = add([x, shortcut], name='g%db%dadd' % (group, block))

        # MSE layer: matrix of squared pairwise distances of the activations
        if id_output:
            x_flatten = Flatten()(x)
            mse = Lambda(
                    pairwise_mse, 
                    name='g{}b{}add_mse'.format(group, block))(x_flatten)
            daug_inv = Lambda(
                    lambda x: K.tile(K.expand_dims(x, axis=2), (1, 1, 2)),
                    name='daug_inv_g{}b{}'.format(group, block))(mse)
            class_inv = Lambda(
                    lambda x: K.tile(K.expand_dims(x, axis=2), (1, 1, 2)),
                    name='class_inv_g{}b{}'.format(group, block))(mse)
            mean_inv = Lambda(
                    lambda x: K.expand_dims(K.mean(x, axis=1)),
                    name='mean_inv_g{}b{}'.format(group, block))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        return x, mse_list

    # Build the network architecture
    inputs = Input(shape=image_shape)
    mse_list = []

    # conv1: Initial convolution
    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(stride_conv1, stride_conv1),
               padding='same',
               use_bias=True,
               kernel_initializer='he_normal',
               bias_initializer='zeros',
               kernel_regularizer=l2reg(weight_decay),   
               bias_regularizer=l2reg(weight_decay),
               name='g0b0conv1')(inputs)

    # Group 1:
    for i in range(0, blocks_per_group):
        n_filters = 16 * widening_factor
        x, mse_list = residual_block(x, mse_list, dropout, weight_decay, 
                                     batch_norm, n_filters, subsample_factor=1,
                                     group=1, block=i+1)

    # Group 2:
    for i in range(0, blocks_per_group):
        n_filters = 32 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x, mse_list = residual_block(x, mse_list, dropout, weight_decay, 
                                     batch_norm, n_filters, subsample_factor, 
                                     group=2, block=i+1)

    # Group 3:
    for i in range(0, blocks_per_group):
        n_filters = 64 * widening_factor
        if i == 0:
            subsample_factor = 2
        else:
            subsample_factor = 1
        x, mse_list = residual_block(x, mse_list, dropout, weight_decay, 
                                     batch_norm, n_filters, subsample_factor, 
                                     group=3, block=i+1)

    # Output group
    x = BatchNormalization(axis=3, 
                           epsilon=1.001e-5,
                           gamma_regularizer=l2reg(weight_decay),
                           beta_regularizer=l2reg(weight_decay),
                           name='g4b1logits_bn')(x)
    x = Activation('relu', name='g4b1logits_relu')(x)
    x = AveragePooling2D(pool_size=(8, 8),
                         strides=None,
                         padding='valid',
                         name='g4b1logits_avg')(x)
    x = Flatten()(x)

    # Logits
    logits = Dense(n_classes, 
                   activation='linear', 
                   use_bias=True,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   name='logits')(x)
    
    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[predictions] + mse_list)
    else:
        model = Model(inputs=inputs, outputs=predictions)

    return model


def densenet(image_shape, n_classes, blocks, growth_rate, theta, dropout=None, 
             weight_decay=None, batch_norm=False, architecture='cifar', 
             bottleneck=True, id_output=False):
    """
    Defines the densely coonected network (DenseNet), originally described in
    https://arxiv.org/pdf/1608.06993.pdf It is implemented in a modular way in
    order to allow for more flexibility.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    blocks_per_group : int
        Number of residual blocks per group. Each new group is defined to be
        wider than the previous one (16 * k, 32 * k, 64 * k).

    widening_factor : int
        Factor (k) by which the width of the residual blocks is multiplied. 

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def dense_block(x, n_blocks, name):
        """
        A Dense Block as defined in the paper: a chain of n_blocks densely
        connected DenseNet-B (conv_block) blocks.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        for i in range(n_blocks):
            x = conv_block(x, growth_rate, name=name + 'b{}'.format(i + 1),
                           bottleneck=bottleneck)

        return x


    def transition_block(x, compression, name):
        """
        A transition block as defined in the section "Pooling layers" of the 
        paper: 
            1. BN -> ReLU -> (N * compression) 1x1 conv with stride 1
            2. Average pooling with stride 2

        # Arguments
            x: input tensor.
            compression: float, compression factor at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        x = BatchNormalization(axis=3, 
                               epsilon=1.001e-5,
                               name=name + 'conv1bn')(x)
        x = Activation('relu', name=name + 'conv1relu')(x)
        x = Conv2D(filters=int(K.int_shape(x)[3] * compression), 
                   kernel_size=(1, 1), 
                   strides=(1, 1),
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2reg(weight_decay),   
                   name=name + 'conv1')(x)

        x = AveragePooling2D(pool_size=(2, 2), 
                             strides=2, 
                             name=name + 'avg')(x)

        return x


    def conv_block(x, growth_rate, name, bottleneck):
        """
        A DenseNet-B block as defined in the section "Bottleneck layers" of
        the paper:
            1. BN -> ReLU -> 4 x growth_rate 1x1 conv with stride 1
            2. BN -> ReLU -> growth_rate 3x3 conv with stride 1
            3. Concatenation of input and output to create dense connectivity

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at dense layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        x1 = BatchNormalization(axis=3, 
                                epsilon=1.001e-5,
                                name=name + 'conv1bn')(x)
        x1 = Activation('relu', name=name + 'conv1relu')(x1)

        if bottleneck:
            x1 = Conv2D(filters=4 * growth_rate, 
                        kernel_size=(1, 1), 
                        strides=(1, 1),
                        padding='same',
                        use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2reg(weight_decay),   
                        name=name + 'conv1')(x1)

            x1 = BatchNormalization(axis=3, 
                                    epsilon=1.001e-5,
                                    name=name + 'conv2bn')(x1)
            x1 = Activation('relu', name=name + 'conv2relu')(x1)

        x1 = Conv2D(filters=growth_rate, 
                    kernel_size=(3, 3), 
                    strides=(1, 1),
                    padding='same', 
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2reg(weight_decay),   
                    name=name + 'conv2')(x1)
        if dropout:
            x1 = Dropout(dropout)(x1)

        x = Concatenate(axis=bn_axis, name=name + 'concat')([x, x1])

        return x

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    inputs = Input(shape=image_shape)
    mse_list = []

    if architecture == 'imagenet':
        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = Conv2D(filters=2 * growth_rate, 
                   kernel_size=(7, 7), 
                   strides=(2, 2), 
                   use_bias=False, 
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2reg(weight_decay),   
                   name='conv1')(x)
        x = BatchNormalization(axis=3, 
                               epsilon=1.001e-5,
                               name='conv1bn')(x)
        x = Activation('relu', name='conv1relu')(x)

        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2, name='pool1')(x)
    elif architecture == 'cifar':
        if bottleneck:
            n_filters = 2 * growth_rate
        else:
            n_filters = 16
        x = Conv2D(filters=n_filters, 
                   kernel_size=(3, 3), 
                   strides=(1, 1), 
                   padding='same',
                   use_bias=False, 
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2reg(weight_decay),   
                   name='conv1')(inputs)
    else:
        raise NotImplementedError

    # MSE layer: matrix of squared pairwise distances of the activations
    if id_output:
        x_flatten = Flatten()(x)
        mse = Lambda(pairwise_mse, 
                     name='conv1mse')(x_flatten)
        daug_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                           (1, 1, 2)),
                          name='daug_inv_conv1')(mse)
        class_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                           (1, 1, 2)),
                           name='class_inv_conv1')(mse)
        mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                           name='mean_inv_conv1')(mse)
        mse_list.append(daug_inv)
        mse_list.append(class_inv)
        mse_list.append(mean_inv)

    # Create Dense Blocks
    for idx_block, b in enumerate(blocks):
        x = dense_block(x, b, name='g{}'.format(idx_block + 1))

        # MSE layer: matrix of squared pairwise distances of the activations
        if id_output:
            x_flatten = Flatten()(x)
            mse = Lambda(pairwise_mse, 
                         name='g{}mse'.format(idx_block + 1))(x_flatten)
            daug_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                              name='daug_inv_b{}'.format(idx_block + 1))(mse)
            class_inv = Lambda(lambda x: K.tile(K.expand_dims(x, axis=2), 
                                               (1, 1, 2)),
                               name='class_inv_b{}'.format(idx_block + 1))(mse)
            mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                               name='mean_inv_b{}'.format(idx_block + 1))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        if idx_block != len(blocks) - 1:
            x = transition_block(x, theta, 
                                 name='g{}trans_'.format(idx_block + 1))

    # Global Average Pooling
    x = BatchNormalization(axis=3, 
                           epsilon=1.001e-5,
                           name='global_avg_bn')(x)
    x = GlobalAveragePooling2D(name='global_avg')(x)

    # Logits
    logits = Dense(n_classes, 
                   activation='linear', 
                   use_bias=True,
                   kernel_initializer='he_normal',
                   bias_initializer='zeros',
                   name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[predictions] + mse_list,
                      name='densenet')
    else:
        model = Model(inputs=inputs, outputs=predictions, name='densenet')

    return model


def lenet(image_shape, n_classes, return_logits=False, dropout=False, 
          weight_decay=None, batch_norm=False, seed=None, id_output=False):
    """
    Defines a version of LeNet to train on MNIST, as described in Madry et al.
    (2017) and also used in Kannan, Kuraking and Goodfellow (2017).

    See: https://github.com/MadryLab/mnist_challenge
    ---

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    return_logits : bool
        If True, the output of the model will be the logits. Otherwise, the 
        output will be the softmax of the logits, as usual

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    inputs = Input(shape=image_shape)
    mse_list = []

    # Convolution 1
    x = Conv2D(filters=32,
               kernel_size=(5, 5),
               strides=(1, 1),
               padding='same',
               activation='linear',   
               use_bias=True,
               kernel_initializer=TruncatedNormal(stddev=0.1, seed=seed),
               bias_initializer=Constant(0.1),
               kernel_regularizer=l2reg(weight_decay),   
               bias_regularizer=l2reg(weight_decay),
               input_shape=image_shape,
               name='conv1')(inputs)
    if batch_norm:
        x = BatchNormalization(axis=3, 
                               epsilon=1.001e-5,
                               gamma_regularizer=l2reg(weight_decay),
                               beta_regularizer=l2reg(weight_decay),
                               name='conv1bn')(x)
    x = Activation('relu', name='conv1relu')(x)    
    x = MaxPooling2D(pool_size=(2, 2), name='conv1pool')(x)

    # MSE layer: matrix of squared pairwise distances of the activations
    if id_output:
        x_flatten = Flatten()(x)
        mse = Lambda(pairwise_mse, name='conv1mse')(x_flatten)
        mse_list.append(mse)

    # Dropout of 25 %
    if dropout:
        x = Dropout(rate=0.25, name='dropout1')(x)

    # Convolution 2
    x = Conv2D(filters=64,
               kernel_size=(5, 5),
               strides=(1, 1),
               padding='same',
               activation='linear',   
               use_bias=True,
               kernel_initializer=TruncatedNormal(stddev=0.1, seed=seed),
               bias_initializer=Constant(0.1),
               kernel_regularizer=l2reg(weight_decay),   
               bias_regularizer=l2reg(weight_decay),
               input_shape=image_shape,
               name='conv2')(x)
    if batch_norm:
        x = BatchNormalization(axis=3, 
                               epsilon=1.001e-5,
                               gamma_regularizer=l2reg(weight_decay),
                               beta_regularizer=l2reg(weight_decay),
                               name='conv2bn')(x)
    x = Activation('relu', name='conv2relu')(x)    
    x = MaxPooling2D(pool_size=(2, 2), name='conv2pool')(x)

    # MSE layer: matrix of squared pairwise distances of the activations
    if id_output:
        x_flatten = Flatten()(x)
        mse = Lambda(pairwise_mse, name='conv2mse')(x_flatten)
        mse_list.append(mse)

    # Dropout of 25 %
    if dropout:
        x = Dropout(rate=0.25, name='dropout2')(x)
    
    # FC layer
    x = Flatten()(x)
    x = Dense(units=1024, 
              activation='linear', 
              kernel_initializer=TruncatedNormal(stddev=0.1, seed=seed),
              bias_initializer=Constant(0.1),
              kernel_regularizer=l2reg(weight_decay),   
              bias_regularizer=l2reg(weight_decay),
              name='fc1')(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, 
                               epsilon=1.001e-5,
                               gamma_regularizer=l2reg(weight_decay),
                               beta_regularizer=l2reg(weight_decay),
                               name='fc1bn')(x)
    x = Activation('relu', name='fc1relu')(x)    
    
    # MSE layer: matrix of squared pairwise distances of the activations
    if id_output:
        mse = Lambda(pairwise_mse, name='conv3mse')(x_flatten)
        mse_list.append(mse)

    # Logits layer (FC)
    logits = Dense(n_classes, 
                   activation='linear', 
                   kernel_initializer=TruncatedNormal(stddev=0.1, seed=seed),
                   bias_initializer=Constant(0.1),
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   name='logits')(x)

    if return_logits:
        output = logits
    else:
        # Softmax
        output = Activation('softmax', name='softmax')(logits)

    if len(mse_list) > 0:
        model = Model(inputs=inputs, outputs=[output] + mse_list)
    else:
        model = Model(inputs=inputs, outputs=output)

    return model


def khonsu(image_shape, n_classes, dropout=0., weight_decay=None,
           batch_norm=False, invariance=True):
    """
    A shallow convolutional neural network

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : float
        The drop rate for Dropout after the convolutional layers.

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    invariance : bool
        If True, a matrix of squared pairwise distances will be computed with
        the activations of each convoltional layer (after ReLU) and appended as
        output of the model.

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, mse_list, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (width) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2reg(weight_decay),   
                   input_shape=image_shape,
                   name='conv{}'.format(n_layer))(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv{}bn'.format(n_layer))(x)

        # ReLU Activation
        x = Activation('relu', name='conv{}relu'.format(n_layer))(x)    

        # MSE layer: matrix of squared pairwise distances of the activations
        if invariance:
            x_flatten = Flatten()(x)
            mse = Lambda(pairwise_mse, 
                         name='conv{}mse'.format(n_layer))(x_flatten)
            daug_inv = Lambda(lambda x: x,
                              name='daug_inv{}'.format(n_layer))(mse)
            class_inv = Lambda(lambda x: x,
                               name='class_inv{}'.format(n_layer))(mse)
            mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                               name='mean_inv{}'.format(n_layer))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        return x, mse_list

    def fc_layer(x, mse_list, units, n_layer):
        """
        Defines a fully connected block, formed by a fully connected layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        units : int
            Number of neurons in the layer

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Fully connected layer
        x = Dense(units=units, 
                  activation='linear', 
                  use_bias=True, 
                  kernel_initializer='he_normal', 
                  bias_initializer='zeros',
                  name='fc{}'.format(n_layer))(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=1, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='fc{}bn'.format(n_layer))(x)

        # ReLU Activation
        x = Activation('relu', name='fc{}relu'.format(n_layer))(x)    

        # MSE layer: matrix of squared pairwise distances of the activations
        if invariance:
            mse = Lambda(pairwise_mse, 
                         name='fc{}mse'.format(n_layer))(x)
            daug_inv = Lambda(lambda x: x,
                              name='daug_inv{}'.format(n_layer))(mse)
            class_inv = Lambda(lambda x: x,
                               name='class_inv{}'.format(n_layer))(mse)
            mean_inv = Lambda(lambda x: K.expand_dims(K.mean(x, axis=1)),
                               name='mean_inv{}'.format(n_layer))(mse)
            mse_list.append(daug_inv)
            mse_list.append(class_inv)
            mse_list.append(mean_inv)

        return x, mse_list

    inputs = Input(shape=image_shape, name='input')
    mse_list = []

    # Conv. layer: 100 filters, stride 2
    x, mse_list = conv_layer(inputs, mse_list, filters=100, kernel=3, stride=2, 
                        padding='same', n_layer=1)

    if dropout > 0.:
        x = Dropout(rate=dropout, name='dropout1')(x)

    # Conv. layer: 75 filters, stride 2
    x, mse_list = conv_layer(x, mse_list, filters=75, kernel=3, stride=2, 
                        padding='same', n_layer=2)

    if dropout > 0.:
        x = Dropout(rate=dropout, name='dropout2')(x)

    # Conv. layer: 50 filters, stride 2
    x, mse_list = conv_layer(x, mse_list, filters=50, kernel=3, stride=2, 
                        padding='same', n_layer=3)

    if dropout > 0.:
        x = Dropout(rate=dropout, name='dropout3')(x)

    # Conv. layer: 25 filters, stride 2
    x, mse_list = conv_layer(x, mse_list, filters=25, kernel=3, stride=2, 
                        padding='same', n_layer=4)

    if dropout > 0.:
        x = Dropout(rate=dropout, name='dropout4')(x)

    # Dense Layer
    x = Flatten(name='flatten_conv_to_fc')(x)
    x, mse_list = fc_layer(x, mse_list, n_classes, n_layer=5)

    output_inv = Lambda(lambda x: x, name='output_inv')(x)
    input_cat = Input(shape=(K.int_shape(output_inv)[-1], ), name='input_cat')

    # Logits
    logits = Dense(n_classes, activation='linear', use_bias=True, 
                   kernel_initializer='he_normal', bias_initializer='zeros',
                   name='logits')

    logits_inv = logits(output_inv)
    logits_cat = logits(input_cat)

    # Softmax
    softmax = Activation('softmax', name='softmax')
    predictions_inv = softmax(logits_inv)
    predictions_cat = softmax(logits_cat)

    if len(mse_list) == 0:
        return Model(inputs=inputs, outputs=predictions_inv)
    else:

        # Invariance model
        model_inv = Model(inputs=inputs, 
                          outputs=[output_inv, predictions_inv] + mse_list,
                          name='model_inv')

        # Categorization model
        model_cat = Model(inputs=input_cat, 
                          outputs=predictions_cat,
                          name='model_cat')

        return [model_inv, model_cat]


def pairwise_mse(x):
    # See https://stackoverflow.com/a/37040451
    # Possibly more efficient, sparse solution:
    # https://github.com/keras-team/keras/issues/7065
    r = K.reshape(K.sum(K.square(x), axis=1), (-1, 1))
    return (r - 2 * K.dot(x, K.transpose(x)) + K.transpose(r)) \
           / K.int_shape(x)[-1]


def l2reg(wd):
    """
    Defines the regularizer for the kernel and bias parameters. It can be
    either None or L2 (weight decay, if the coefficient is divided by 2. [see
    https://bbabenko.github.io/weight-decay/]).

    Parameters
    ----------
    wd : float
        L2 regularization factor.

    Returns
    -------
    l2 : function
        Regularization function or None
    """
    if wd is None:
        return None
    else:
        return l2(wd / 2.)

