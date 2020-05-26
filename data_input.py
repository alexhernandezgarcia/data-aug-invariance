"""
Routine for decoding the handling HDF5 files and providing the input images
to the trainer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from image_data_generator import ImageDataGeneratorExt

import numpy as np

import yaml
import dask.array as da
import h5py
import time
from os import remove

from tqdm import tqdm


class batch_generator(object):

    def __init__(self, image_gen, images, labels, batch_size, aug_per_im, 
                 shuffle, seed=None, n_inv_layers=0):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.aug_per_im = aug_per_im
        self.shuffle = shuffle
        self.seed = seed
        self.n_inv_layers = n_inv_layers
        self.n_images = images.shape[0]
        self.n_batches = int(np.ceil(self.n_images / float(self.batch_size)))
        if isinstance(images, np.ndarray):
            self.image_gen = image_gen.flow(
                    self.images, self.labels, batch_size=self.batch_size,
                    aug_per_im=self.aug_per_im, shuffle=self.shuffle,
                    seed_shuffle=self.seed)
        elif isinstance(images, da.core.Array):
            self.image_gen = image_gen.flow_dask(
                    self.images, self.labels, batch_size=self.batch_size,
                    aug_per_im=self.aug_per_im, shuffle=self.shuffle,
                    seed_shuffle=self.seed)
        else:
            raise NotImplementedError('The input data must be a Numpy or a '
                                      'Dask array')

    def __call__(self):
        batch = next(self.image_gen)
        if self.n_inv_layers > 0:
            # The list of targets for the invariance outputs are:
            # [daug_invariance_target, class_invariance_target, mean: ones]
            # for each invariance layer
            bs = batch[0].shape[0]
            class_inv_y = np.dot(batch[1][0], batch[1][0].T)
            ones = np.ones([bs, bs], dtype=np.uint8)
            tril = np.tril_indices(bs)
            class_inv_y[tril] = 0
            ones[tril] = 0
#             daug_inv_y = np.stack([batch[1][1], class_inv_y], axis=2)
            daug_inv_y = np.stack([batch[1][1], ones], axis=2)
            class_inv_y = np.stack([class_inv_y, ones], axis=2)
            invariance_targets = [daug_inv_y, 
                                  class_inv_y,
                                  np.ones(batch[0].shape[0], dtype=np.uint8)] \
                                 * self.n_inv_layers
            yield (batch[0],
                   [batch[1][0]] + invariance_targets)
        else:
            yield (batch[0], batch[1][0])


def get_generator(images, **dict_params):
    """
    Initializes the image data generator

    Parameters
    ----------
    images : dask.array
        The array containing the whole set of images. Must be of
        [N, height, width, 3] size.

    dict_params : dict
        Keyword arguments containing the data augmentation parameters

    Yields
    -------
    ImageDataGeneratorExt
        The image generator
    """

    # Initialize data generator
    if 'synthetic' in dict_params:
        image_gen = SyntheticDataGenerator(**dict_params)
    else:
        image_gen = ImageDataGeneratorExt(**dict_params)

    # Compute internal data statistics
    if dict_params['featurewise_center'] | \
       dict_params['featurewise_std_normalization'] | \
       dict_params['zca_whitening']:
        image_gen.fit(images)

    return image_gen


def generate_batches(image_gen, images, labels, batch_size, aug_per_im, shuffle, 
                     seed=None, n_inv_layers=0):
    for batch in image_gen.flow_dask(images, labels,
                                     batch_size=batch_size,
                                     aug_per_im=aug_per_im,
                                     shuffle=shuffle,
                                     seed_shuffle=seed):
        if n_inv_layers > 0:
            # The list of targets for the invariance outputs are:
            # [daug_invariance_target, class_invariance_target, mean: ones]
            # for each invariance layer
            bs = batch[0].shape[0]
            class_inv_y = np.dot(batch[1][0], batch[1][0].T)
            ones = np.ones([bs, bs], dtype=np.uint8)
            tril = np.tril_indices(bs)
            class_inv_y[tril] = 0
            ones[tril] = 0
            daug_inv_y = np.stack([batch[1][1], class_inv_y], axis=2)
            class_inv_y = np.stack([class_inv_y, ones], axis=2)
            invariance_targets = [daug_inv_y, 
                                  class_inv_y,
                                  np.ones(batch[0].shape[0], dtype=np.uint8)] \
                                 * n_inv_layers
            yield (batch[0],
                   [batch[1][0]] + invariance_targets)
        else:
            yield (batch[0], batch[1][0])


def train_val_split(hdf5_file, group_train, group_test, chunk_size, 
                    pct_train=1., pct_val=1., shuffle=False, seed=None, 
                    labels_id='labels'):
    """
    Creates the train/validation split, either by spliting the training set or
    from a specified test set.

    Parameters
    ----------
    hdf5_file : h5py Object
        h5py Object containing the data set

    group_train : str
        Name of the group containing the training set

    group_test : str
        Name of the group containing the test (validation) set. It can be None

    chunk_size : int
        Size of the chunks of the dask arrays

    pct_train : float
        It allows training with a reduced set of the available training data

    pct_val : float
        Percentage of training examples used for validation. Only relevant if
        group_test is None.

    shuffle : bool
        Whether to shuffle before the train/validation split. Only relevant if
        group_test is None.
        
    seed : int
        Seed for the shuffle.

    Returns
    -------
    da_images_tr : dask.array
        Training images
        
    da_images_val : dask.array
        Validation images

    da_labels_tr : dask.array
        Training labels

    da_labels_val: dask.array
        Validation labels

    hdf5_files : list
        Auxiliar HDF5 files, if any
    """
    hdf5_files = []

    if group_test:

        da_images_val, da_labels_val, _ = hdf52dask(hdf5_file, group_test, 
                                                    chunk_size, shuffle=False, 
                                                    seed=None, pct=1.0,
                                                    labels_id=labels_id)  

        da_images_tr, da_labels_tr, hdf5_aux = hdf52dask(hdf5_file, 
                                                         group_train, 
                                                         chunk_size, shuffle, 
                                                         seed, pct_train,
                                                         labels_id=labels_id)  
        if hdf5_aux:
            hdf5_files.append(hdf5_aux)

    else:

        da_images_val, da_labels_val, hdf5_aux = hdf52dask(hdf5_file, 
                                                           group_train, 
                                                           chunk_size, shuffle, 
                                                           seed, 
                                                           pct=(1.0 - pct_val),
                                                           from_tail=True,
                                                           labels_id=labels_id)  
        if hdf5_aux:
            hdf5_files.append(hdf5_aux)

        da_images_tr, da_labels_tr, hdf5_aux = hdf52dask(hdf5_file, 
                                                         group_train, 
                                                         chunk_size, shuffle, 
                                                         seed, 
                                                         pct=(1.0 - pct_val),
                                                         labels_id=labels_id)  
        if hdf5_aux:
            hdf5_files.append(hdf5_aux)

    return da_images_tr, da_images_val, da_labels_tr, da_labels_val, hdf5_files


def dataset_characteristics(hdf5_file, group_name, labels_id='labels'):
    """
    Counts the number of examples, the number of classes and the shape of the
    images in a data set from an HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File
        h5py File object with read permission

    group_name : str
        The name of the data set

    Returns
    -------
    num_examples : int
        The number of examples in the TFRecord files

    num_classes : int
        The number of classes in the TFRecord files

    height : int
        The height of one image

    width : int
        The width of one image

    depth : int
        The depth of one image
    """

    data = hdf5_file[group_name]
    images = data['data']
    labels = data[labels_id]

    num_examples = images.shape[0]
    num_classes = labels.shape[1]
    if images.ndim > 2:
        height = images.shape[1]
        width = images.shape[2]
        if images.ndim > 3:
            depth = images.shape[3]
        else:
            depth = 1
        image_shape = (height, width, depth)
    else:
        image_shape = (images.shape[1], )

    print('Number of available examples: %d' % num_examples)
    print('Number of classes: %d' % num_classes)
    print('Shape of input images: {}'.format(image_shape))
    print('')

#     return num_examples, num_classes, height, width, depth
    return num_examples, num_classes, image_shape


def hdf52dask(hdf5_file, group=None, chunk_size=None, shuffle=False, seed=None,
              pct=1.0, from_tail=False, labels_id='labels'):
    """
    Converts an HDF5 matrix data set into a dask array

    Parameters
    ----------
    hdf5_file : h5py Object
        h5py Object containing the data set

    group : str
        Name of the group containing the data set

    chunk_size : int
        Size of the chunks of the dask arrays

    shuffle : bool
        Whether to shuffle the elements before converting into dask

    seed : int
        Seed for the shuffle

    pct : float
        Percentage of examples to keep in the final arrays

    from_tail : bool
        Whether the elements should be retrieved from the tail (end) of the 
        array. This allows for train/val splitting within the same HDF5 file.

    Returns
    -------
    da_images : Dask array
        The dask array containing the images

    da_labels : Dask array
        The dask array containing the labels

    hdf5_aux : h5py Object or None
        The auxiliary HDF5 file for shuffling, or None if no shuffling is
        applied.
    """

    if (pct <= 0.0) | (pct > 1.0):
        raise ValueError('The percentage of examples must be larger than 0. '
                         'and smaller than or equal to 1.')

    # Read data from HDF5 file
    if group:
        data = hdf5_file[group]
    else:
        data = hdf5_file
    images = data['data']
    labels = data[labels_id]

    # Reduce and/or shuffle the data set according to pct and shuffle 
    n = int(np.floor(pct * images.shape[0]))
    if shuffle:
        if seed:
            np.random.seed(seed)
        if from_tail:
            idx = np.random.permutation(images.shape[0])[n:]
        else:
            idx = np.random.permutation(images.shape[0])[:n]
        images, labels, hdf5_aux = _shuffle_data(images, labels, idx,
                                                 labels_id=labels_id)
    else:
        if from_tail:
            images = images[n:]
            labels = labels[n:]
        else:
            images = images[:n]
            labels = labels[:n]
        hdf5_aux = []

    # Create dask arrays
    da_images = da.from_array(images, chunks=(chunk_size,) + images.shape[1:])
    da_labels = da.from_array(labels, chunks=(chunk_size, labels.shape[1]))

    return da_images, da_labels, hdf5_aux


def subsample_data(images, labels, pct, chunk_size=None, 
                   shuffle=False, seed=None, labels_id='labels'):
    hdf5_aux = []
    if pct < 1.:
        # Store the data into an auxiliary HDF5 file
        filename = 'hdf5_aux_{}'.format(time.time())
        da.to_hdf5(filename, {'data': images, labels_id: labels})

        # Read HDF5
        hdf5_aux1 = h5py.File(filename, 'r')

        images, labels, hdf5_aux2 = hdf52dask(hdf5_aux1, group=None,
                chunk_size=chunk_size, shuffle=shuffle, seed=seed, pct=pct,
                labels_id=labels_id)

        if hdf5_aux2:
            hdf5_aux.extend([hdf5_aux1, hdf5_aux2])
        else:
            hdf5_aux.extend([hdf5_aux1])

    return images, labels, hdf5_aux



def _shuffle_data(images_orig, labels_orig, idx, labels_id='labels'):
    """
    Shuffles the images and labels of an HDF5 file, according to a given vector
    of indices, by creating a new auxiliary HDF5 file.
    
    Parameters
    ----------
    images_orig : h5py dataset
        The original images

    labels_orig : h5py dataset
        The original labels

    idx : array
        An array of (random) indices used to re-arrange the elements of the
        HDF5 data
        
    Returns
    -------
    images : h5py dataset
        The shuffled images

    labels : h5py dataset
        The shuffled labels

    hdf5_aux : h5py Object
        The new auxiliary HDF5 file
    """

    filename = 'hdf5_aux_{}'.format(time.time())
    hdf5_aux = h5py.File(filename, 'w')
    images = hdf5_aux.create_dataset('data', 
                                     shape=(idx.shape[0],) + 
                                            images_orig.shape[1:], 
                                     dtype=np.uint8)
    labels = hdf5_aux.create_dataset(labels_id, 
                                     shape=(idx.shape[0], 
                                            labels_orig.shape[1]), 
                                     dtype=np.uint8)
    print('\nShuffling data...')
    for i, idx in enumerate(tqdm(idx)):
        images[i] = images_orig[idx]
        labels[i] = labels_orig[idx]

    hdf5_aux.close()
    hdf5_aux = h5py.File(filename, 'r')
    images = hdf5_aux['data']
    labels = hdf5_aux[labels_id]

    return images, labels, hdf5_aux


def create_control_dataset(images, labels, daug_params, nodaug_params, 
                           n_per_image, n_per_class, chunk_size=None, 
                           seed=None):

    if seed:
        np.random.seed(seed)

    labels_np = np.asarray(labels)
    labels_int = [np.where(label == 1)[0][0] for label in labels_np]
    n_classes = len(np.unique(labels_int))
    indices = np.random.permutation(len(labels_int))

    cum_cl = np.zeros(n_classes, dtype=int)
    images_sel = np.zeros([n_classes, n_per_class], dtype=int)

    for idx in indices:
        if cum_cl[labels_int[idx]] < n_per_class:
            images_sel[labels_int[idx], cum_cl[labels_int[idx]]] = idx
            cum_cl[labels_int[idx]] += 1

    dataset_images = []
    dataset_labels = []

    for cl in range(n_classes):
        for img in range(n_per_class):

            image = da.from_array(np.expand_dims(
                images[images_sel[cl, img]], axis=0))
            label = da.from_array(np.expand_dims(
                labels[images_sel[cl, img]], axis=0))

            image_gen_daug = get_generator(image, **daug_params)
            batch_gen_daug = batch_generator(image_gen_daug, image, label, 
                                             batch_size=1,
                                             aug_per_im=n_per_image - 1, 
                                             shuffle=False)
            image_gen_nodaug = get_generator(image, **nodaug_params)
            batch_gen_nodaug = batch_generator(image_gen_nodaug, image, label, 
                                               batch_size=1,
                                               aug_per_im=1, 
                                               shuffle=False)

            batch_images, batch_labels = next(batch_gen_nodaug())
            dataset_images.append(da.from_array(batch_images))
            dataset_labels.append(da.from_array(batch_labels))

            batch_images, batch_labels = next(batch_gen_daug())
            dataset_images.append(da.from_array(batch_images))
            dataset_labels.append(da.from_array(batch_labels))

    dataset_images = da.concatenate(dataset_images, axis=0)
    dataset_labels = da.concatenate(dataset_labels, axis=0)
    if chunk_size is None:
        chunk_size = dataset_images.shape[0]
    dataset_images = da.rechunk(dataset_images, 
                                (chunk_size, ) + (dataset_images.shape[1:]))
    dataset_labels = da.rechunk(dataset_labels, 
                                (chunk_size, ) + (dataset_labels.shape[1:]))

    # Convert back the images into uint8 and [0, 255]
    # Note though that it would be better to avoid this and implement 
    # additional functionality in image.py to avoid the normalization
    dataset_images *= 255
    dataset_images = dataset_images.astype(np.uint8)

    return dataset_images, dataset_labels


def validation_image_params(base_config_file=
        '/mnt/data/alex/git/research/projects/daug/daug_schemes/nodaug.yml', 
        **params_dict):
    """
    Sets up the image configuration parameters for validation according to the
    training parameters in terms of normalization, color space, etc. 

    Parameters
    ----------
    base_config_file : str
        Path to the base configuration file, which should contain the default
        values of all parameters

    params_dict : dict
        Dictionary with the training image parameters

    Returns
    -------
    val_config : dict
        The validation parameters dictionary
    """
    with open(base_config_file, 'r') as yml_file:
        val_config = yaml.load(yml_file, Loader=yaml.FullLoader)

    # Standardization parameters
    val_config['featurewise_center'] = params_dict['featurewise_center']
    val_config['samplewise_center'] = params_dict['samplewise_center']
    val_config['featurewise_std_normalization'] = \
        params_dict['featurewise_std_normalization']
    val_config['samplewise_std_normalization'] = \
        params_dict['samplewise_std_normalization']
    val_config['zca_whitening'] = params_dict['zca_whitening']

    # Crop parameters
    if (params_dict['do_random_crop'] | params_dict['do_central_crop']) & \
       (params_dict['crop_size'] is not None):
        val_config['do_random_crop'] = False
        val_config['do_central_crop'] = True
        val_config['crop_size'] = params_dict['crop_size']

    # Color space
    val_config['color_space'] = params_dict['color_space']

    return val_config

