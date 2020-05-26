"""
DaskArrayIterator: extended from NumpyArrayIterator as in
https://github.com/keras-team/keras-preprocessing/blob/master/
keras_preprocessing/image/numpy_array_iterator.py
but with extended functionality to yield data from a dask array and provide
functionality to support data augmentation invariance.
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

import keras.backend as K
from keras_preprocessing.image.iterator import Iterator
from keras_preprocessing.image.utils import array_to_img

import warnings
import os


class DaskArrayIterator(Iterator):
    """
    Iterator yielding data from a dask array.

    Arguments
    ---------
    x : dask.array
        dask array of input data

    y : dask.array
        dask array of targets data

    target_shape : list
        Shape of the images. Necessary if cropping is performed.

    image_data_generator : Instance of `ImageDataGenerator`
        To use for random transformations and normalization.

    batch_size : int
       Size of a batch.

    aug_per_im : int
        Number of transformations per image

    shuffle : bool
        Whether to shuffle the data between epochs.

    seed : int
        Random seed for data shuffling.

    data_format : str
        One of `channels_first`, `channels_last`.

    save_to_dir : str
        Optional directory where to save the pictures being yielded, in a
        viewable format. This is useful for visualizing the random
        transformations being applied, for debugging purposes.

    save_prefix : str
        Prefix to use for saving sample images (if `save_to_dir` is set).

    save_format : str
        Format to use for saving sample images (if `save_to_dir` is set).
    """

    def __new__(cls, *args, **kwargs):
        try:
            from tensorflow.keras.utils import Sequence as TFSequence
            if TFSequence not in cls.__bases__:
                cls.__bases__ = cls.__bases__ + (TFSequence,)
        except ImportError:
            pass
        return super(DaskArrayIterator, cls).__new__(cls)

    def __init__(self,
                 x,
                 y,
                 target_shape,
                 image_data_generator,
                 batch_size=32,
                 aug_per_im=1,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 ignore_class_split=False,
                 dtype=K.floatx()):
        # Note that most lines are adapted from the __init__ function of
        # NumpyArrayIterator. Importantly, np.asarray(x) (or on y) is never
        # performed here, since the memory could be filled up.
        self.dtype = dtype
        if (type(x) is tuple) or (type(x) is list):
            if type(x[1]) is not list:
                x_misc = [x[1]]
            else:
                x_misc = [xx for xx in x[1]]
            x = x[0]
            for xx in x_misc:
                if len(x) != len(xx):
                    raise ValueError(
                        'All of the arrays in `x` '
                        'should have the same length. '
                        'Found a pair with: len(x[0]) = %s, len(x[?]) = %s' %
                        (len(x), len(xx)))
        else:
            x_misc = []

        if (type(y) is tuple) or (type(y) is list):
            if type(y[1]) is not list:
                y_misc = [y[1]]
            else:
                y_misc = [yy for yy in y[1]]
            y = y[0]
            for yy in y_misc:
                if len(y) != len(yy):
                    raise ValueError(
                        'All of the arrays in `y` '
                        'should have the same length. '
                        'Found a pair with: len(y[0]) = %s, len(y[?]) = %s' %
                        (len(y), len(yy)))
        else:
            y_misc = []

        if y is not None and len(x) != len(y):
            raise ValueError('x (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (x.shape, y.shape))
        if sample_weight is not None and len(x) != len(sample_weight):
            raise ValueError('`x` (images tensor) and `sample_weight` '
                             'should have the same length. '
                             'Found: x.shape = %s, sample_weight.shape = %s' %
                             (x.shape, sample_weight.shape))
        if subset is not None:
            if subset not in {'training', 'validation'}:
                raise ValueError('Invalid subset name:', subset,
                                 '; expected "training" or "validation".')
            split_idx = int(len(x) * image_data_generator._validation_split)

            if (y is not None and not ignore_class_split and not
               np.array_equal(
                   da.unique(y[:split_idx]).compute(),
                   da.unique(y[split_idx:])).compute()):
                raise ValueError('Training and validation subsets '
                                 'have different number of classes after '
                                 'the split. If your numpy arrays are '
                                 'sorted by the label, you might want '
                                 'to shuffle them.')

            if subset == 'validation':
                x = x[:split_idx]
                x_misc = [xx[:split_idx] for xx in x_misc]
                if y is not None:
                    y = y[:split_idx]
            else:
                x = x[split_idx:]
                x_misc = [xx[split_idx:] for xx in x_misc]
                if y is not None:
                    y = y[split_idx:]

        # Define the dask arrays and the chunk size. The size is assumed to be
        # the 0th dimension of the chunks and the other dimensions should be
        # equal to x.shape[1:]
        self.x_dask = x
        self.x_misc_dask = x_misc
        self.chunk_size = self.x_dask.chunks[0][0]

        # First chunk
        self.x = np.asarray(self.x_dask[:self.chunk_size], dtype=self.dtype)
        self.x_misc = [np.asarray(xx[:self.chunk_size]) for xx in
                         self.x_misc_dask]
        self.chunk_index = 0

        if y is not None:
            self.y_dask = y
            self.y = np.asarray(self.y_dask[:self.chunk_size])
            self.y_misc_dask = y_misc
            self.y_misc = [np.asarray(yy[:self.chunk_size]) for yy in
                             self.y_misc_dask]
        else:
            self.y_dask = None
            self.y = None
            self.y_misc_dask = None
            self.y_misc = None

        if sample_weight is not None:
            self.sample_weight_dask = sample_weight
            self.sample_weight = np.asarray(
                self.sample_weight_dask[:self.chunk_size])
        else:
            self.sample_weight_dask = None
            self.sample_weight = None

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                          '(channels on axis ' + str(channels_axis) +
                          '), i.e. expected either 1, 3, or 4 '
                          'channels on axis ' + str(channels_axis) + '. '
                          'However, it was passed an array with shape ' +
                          str(self.x.shape) + ' (' +
                          str(self.x.shape[channels_axis]) + ' channels).')

        self.n_aug = aug_per_im
        self.n_images = self.x_dask.shape[0]
        self.target_shape = target_shape
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        super(DaskArrayIterator, self).__init__(self.chunk_size,
                                                batch_size,
                                                shuffle,
                                                seed)

    # Overwritten to support:
    #   - image cropping
    #   - list of labels y
    #   - in-batch augmentation
    def _get_batches_of_transformed_samples(self, index_array):
        # IMPORTANT: the next line is changed with respect to the original
        # keras implementation
        # Change: self.target_shape <-- list(self.x.shape)[1:]
        # Additionally: support for more than one augmentation per image
        batch_x = np.zeros(tuple([len(index_array) * self.n_aug] \
                                 + self.target_shape), dtype=self.dtype)
        batch_y_id = np.zeros([len(index_array) * self.n_aug] * 2,
                              dtype=np.uint8)
        for i, j in enumerate(index_array):
            batch_y_id[i * self.n_aug:i * self.n_aug + self.n_aug,
                       i * self.n_aug:i * self.n_aug + self.n_aug] = 1
            for k in range(self.n_aug):
                x = self.x[j]
                params = self.image_data_generator.get_random_transform(
                                                    x.shape)
                x = self.image_data_generator.apply_transform(
                    x.astype(self.dtype), params)
                x = self.image_data_generator.standardize(x)
                batch_x[i * self.n_aug + k] = x
        if self.n_aug > 1:
            batch_y_id[np.tril_indices(batch_y_id.shape[0])] = 0

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        # Re-shuffle, in order to avoid contiguous augmented images
        if (self.n_aug > 1) & (self.shuffle):
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            idx = np.random.permutation(batch_x.shape[0])
        else:
            idx = np.arange(batch_x.shape[0])

        batch_x = batch_x[idx]
        batch_x_miscs = [xx[index_array][idx] for xx in self.x_misc]

        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)

        if self.y is None:
            return output[0]

        batch_y_cat = np.repeat(self.y[index_array], self.n_aug, axis=0)[idx]
        batch_y_id = batch_y_id[idx][:, idx]
        batch_y = [batch_y_cat, batch_y_id]
        batch_y_miscs = [yy[indey_array][idx] for yy in self.y_misc]
        output += (batch_y if batch_y_miscs == []
                  else batch_y + batch_y_miscs,)

        if self.sample_weight is not None:
            output += (np.repeat(self.sample_weight[index_array], self.n_aug,
                axis=0)[idx],)
        return output

    # Overwritten to implement dask chunks functionality
    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1

        if self.index_array is None:
            self._set_index_array()

        # Check new chunk
        if idx == 0:

            current_chunk_index = (self.chunk_index * self.n) % self.n_images
            self.x = np.asarray(self.x_dask[current_chunk_index:
                                            current_chunk_index + self.n],
                                dtype=self.dtype)
            self.x_misc = [np.asarray(xx[current_chunk_index:
                                         current_chunk_index + self.n])
                           for xx in self.x_misc_dask]
            if self.y is not None:
                self.y = np.asarray(self.y_dask[current_chunk_index:
                                                current_chunk_index + self.n])
                self.y_misc = [np.asarray(yy[current_chunk_index:
                                             current_chunk_index + self.n])
                               for yy in self.y_misc_dask]
            self.n = self.x.shape[0]
            self._set_index_array()

        # Check last chunk
        if idx == len(self) - 1:
            self.chunk_index += 1

        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]

        return self._get_batches_of_transformed_samples(index_array)

    # Just copied from NumpyArrayIterator
    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    # Overwritten to implement dask chunks functionality
    def on_epoch_end(self):
        if (self.chunk_index * self.chunk_size) >= self.n_images:
            self.reset()

    # Overwritten to implement dask chunks functionality
    def reset(self):
        self.batch_index = 0
        self.chunk_index = 0
        self.x = np.asarray(self.x_dask[:self.chunk_size], dtype=self.dtype)
        self.x_misc = [np.asarray(xx[:self.chunk_size]) for xx in
                         self.x_misc_dask]
        if self.y is not None:
            self.y = np.asarray(self.y_dask[:self.chunk_size])
            self.y_misc = [np.asarray(yy[:self.chunk_size]) for yy in
                             self.y_misc_dask]
        self.n = self.x.shape[0]
        self._set_index_array()

    # Overwritten to implement dask chunks functionality
    def _flow_index(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        # Set variables for the first batch of the first chunk
        self.reset()

        while 1:

            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)

            # Reset 'current' indices in case of initialization
            if self.batch_index == 0:
                current_batch_index = 0
            if self.chunk_index == 0:
                current_chunk_index = 0

            # Check end of chunk
            if self.n <= current_batch_index + self.batch_size:
                self.batch_index = 0

                # Check end of array
                if self.n_images <= current_chunk_index + self.n:
                    self.chunk_index = 0
                    self.n = self.chunk_size
                else:
                    self.chunk_index += 1

                # Define new chunk
                current_chunk_index = (self.chunk_index * self.n) \
                                      % self.n_images
                self.x = np.asarray(self.x_dask[current_chunk_index:
                                                current_chunk_index + self.n],
                                    dtype=self.dtype)
                self.x_misc = [np.asarray(xx[current_chunk_index:
                                             current_chunk_index + self.n])
                               for xx in self.x_misc_dask]
                if self.y is not None:
                    self.y = np.asarray(
                        self.y_dask[current_chunk_index:
                                    current_chunk_index + self.n])
                    self.y_misc = [np.asarray(yy[current_chunk_index:
                                                 current_chunk_index + self.n])
                                   for yy in self.y_misc_dask]
                self.n = self.x.shape[0]
                self._set_index_array()

            current_batch_index = (self.batch_index * self.batch_size) % self.n
            self.batch_index += 1
            self.total_batches_seen += 1

            yield self.index_array[current_batch_index:
                                   current_batch_index + self.batch_size]

