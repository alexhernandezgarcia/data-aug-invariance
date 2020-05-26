"""
Extension of Keras' ImageDataGenerator, from 
https://github.com/keras-team/keras-preprocessing/blob/master/
keras_preprocessing/image/image_data_generator.py

It incorporates additional transformations and crucially the use of 
DaskArrayIterator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import os

import numpy as np

try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
except ImportError:
    scipy = None

from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.affine_transformations import (
        apply_affine_transform, apply_brightness_shift, apply_channel_shift,
        flip_axis)

from image_transformations_ext import (
        truncnorm_cent_2std, gaussian_noise, crop_image, adjust_brightness,
        adjust_contrast, rgb2lab, lab_noise, cutout)

from dask_array_iterator import DaskArrayIterator
import keras.backend as K

class ImageDataGeneratorExt(ImageDataGenerator):
    """
    Extension of Keras' ImageDataGenerator with additional functionality:
        - A larger set of transformations
        - Independent random seeds setting
        - DaskArrayIterator

    Parameters
    ---------
    color_space : str
        'rgb' or 'lab'

    mean_gaussian_std : float
        Standard deviation of Gaussian noise

    do_random_crop : bool

    do_central_crop : bool

    crop_size : list
        Target size of of the crop.
        See: crop_image()

    brightness : float
        Maximum delta for brightness adjustment
        See: adjust_brightness()

    contrast : float list
        Minimum and maximum gamma for contrast adjustment
        See: adjust_contrast()

    lab_noise_range : float
        Maximum peak of the noise in the Lab space
        See: lab_noise()

    train_lab : int
        If True, training images are converted into the Lab space

    rand_distr : str
        Distribution for the augmentation parameters. Can be 'normal' or
        'uniform'
        
    seed_daug : int
        Random seed for the image transformations.
    """

    def __init__(self,
                 color_space='rgb',
                 mean_gaussian_std=0.,
                 do_random_crop=False,
                 do_central_crop=False,
                 crop_size=None,
                 brightness=0.,
                 contrast=[0., 0.],
                 cutout=[0., 0.],
                 lab_noise_range=0.,
                 train_lab=False,
                 rand_distr='normal',
                 seed_daug=None,
                 *args,
                 **kwargs):
        super(ImageDataGeneratorExt, self).__init__(*args, **kwargs)
        self.color_space = color_space
        self.mean_gaussian_std = mean_gaussian_std
        self.random_crop = do_random_crop
        self.central_crop = do_central_crop
        self.crop_size = crop_size
        self.brightness = brightness
        self.contrast = contrast
        self.cutout = cutout
        self.lab_noise_range = lab_noise_range
        self.train_lab = train_lab
        self.rand_distr = rand_distr
        self.seed_daug=seed_daug

        # Image cropping
        if (self.random_crop or self.central_crop) and \
           (self.crop_size is not None):
            self.target_shape = self.crop_size
        else:
            self.target_shape = None

    def flow_dask(self, x, y=None, batch_size=32, aug_per_im=1, shuffle=True,
                  sample_weight=None, seed_shuffle=None, save_to_dir=None,
                  save_prefix='', save_format='png', subset=None,
                  dtype=K.floatx()):

        # Image cropping
        if self.target_shape is None:
            target_shape = list(x.shape)[1:]
        else:
            target_shape = self.target_shape

        # Dask array iterator
        return DaskArrayIterator(
            x, y, target_shape, self, 
            batch_size=batch_size,
            aug_per_im=aug_per_im,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed_shuffle,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            dtype=dtype)

    # Overwritten to add new transformations and additional functionality, such
    # as normal sampling of the random parameters, and different behaviour
    # regarding the random seeds
    def get_random_transform(self, img_shape, seed=None):
        """
        Generates random parameters for a transformation.

        # Parameters
            img_shape: int tuple.
                Shape of the images

            seed: int
                Random seed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        # Modified wrt to the original implementation to perform a
        # deterministic set of different transformations to the subsequent
        # images rather than the same transformation to all images.
        if seed is not None:
            np.random.seed(seed)
        elif self.seed_daug is not None:
            np.random.seed(self.seed_daug)
            if self.seed_daug < np.iinfo(type(self.seed_daug)).max:
                self.seed_daug +=1
            else:
                self.seed_daug = 1
        else:
            pass

        # Extended to the Keras implementation: functionality to support other
        # sampling distributions for the parameters, apart from uniform.

        # Rotation
        if self.rotation_range:
            if self.rand_distr == 'uniform':
                theta = np.random.uniform(-self.rotation_range, 
                                          self.rotation_range)
            elif self.rand_distr == 'normal':
                theta = truncnorm_cent_2std(-self.rotation_range, 
                                           self.rotation_range)
            elif self.rand_distr == 'extremes':
                theta = np.random.choice([-self.rotation_range,
                                          self.rotation_range])
            else:
                raise NotImplementedError()
        else:
            theta = 0

        # Height shift range
        if self.height_shift_range:
            try:
                if isinstance(self.height_shift_range, 
                              (list, tuple, np.ndarray)):  # 1-D array-like
                    tx = np.random.choice(self.height_shift_range)
                    tx *= np.random.choice([-1, 1])
                else:
                    raise ValueError
            except ValueError:  # floating point
                if self.rand_distr == 'uniform':
                    tx = np.random.uniform(-self.height_shift_range, 
                                           self.height_shift_range)
                elif self.rand_distr == 'normal':
                    tx = truncnorm_cent_2std(-self.height_shift_range, 
                                             self.height_shift_range)
                elif self.rand_distr == 'extremes':
                    tx = np.random.choice([-self.height_shift_range,
                                           self.height_shift_range])
                else:
                    raise NotImplementedError()
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        # Width shift range
        if self.width_shift_range:
            try:  # 1-D array-like or int
                if isinstance(self.width_shift_range, 
                              (list, tuple, np.ndarray)):  # 1-D array-like
                    ty = np.random.choice(self.width_shift_range)
                    ty *= np.random.choice([-1, 1])
                else:
                    raise ValueError
            except ValueError:  # floating point
                if self.rand_distr == 'uniform':
                    ty = np.random.uniform(-self.width_shift_range, 
                                           self.width_shift_range)
                elif self.rand_distr == 'normal':
                    ty = truncnorm_cent_2std(-self.width_shift_range, 
                                             self.width_shift_range)
                elif self.rand_distr == 'extremes':
                    ty = np.random.choice([-self.width_shift_range,
                                           self.width_shift_range])
                else:
                    raise NotImplementedError()
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        # Shear range
        if self.shear_range:
            if self.rand_distr == 'uniform':
                shear = np.random.uniform(-self.shear_range, self.shear_range)
            elif self.rand_distr == 'normal':
                shear = truncnorm_cent_2std(-self.shear_range,
                                            self.shear_range)
            elif self.rand_distr == 'extremes':
                shear = np.random.choice([-self.shear_range,
                                          self.shear_range])
            else:
                raise NotImplementedError()
        else:
            shear = 0

        # Zoom (scaling)
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            if self.rand_distr == 'uniform':
                zx, zy = np.random.uniform(self.zoom_range[0], 
                                           self.zoom_range[1], 2)
            elif self.rand_distr == 'normal':
                zx, zy = truncnorm_cent_2std(self.zoom_range[0], 
                                             self.zoom_range[1], 2)
            elif self.rand_distr == 'extremes':
                zx, zy = np.random.choice([self.zoom_range[0],
                                           self.zoom_range[1]], 2)
            else:
                raise NotImplementedError()

        # Horizontal and vertical flips
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        # Channel shift range
        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            if self.rand_distr == 'uniform':
                channel_shift_intensity = np.random.uniform(
                        -self.channel_shift_range,
                        self.channel_shift_range)
            elif self.rand_distr == 'normal':
                channel_shift_intensity = truncnorm_cent_2std(
                        -self.channel_shift_range,
                        self.channel_shift_range)
            elif self.rand_distr == 'extremes':
                channel_shift_intensity = np.random.choice(
                        [-self.channel_shift_range,
                        self.channel_shift_range])
            else:
                raise NotImplementedError()

        # Gaussian noise (not in original Keras impl.)
        if self.mean_gaussian_std > 0:
            if self.rand_distr == 'uniform':
                gaussian_noise_std = np.random.uniform(
                        0., 2. * self.mean_gaussian_std)
            elif self.rand_distr == 'normal':
                gaussian_noise_std = truncnorm_cent_2std(
                        0., 2. * self.mean_gaussian_std)
            elif self.rand_distr == 'extremes':
                gaussian_noise_std = self.mean_gaussian_std
            elif self.rand_distr == 'triangular':
                mode = (0. + self.mean_gaussian_std) / 2.
                gaussian_noise_std = np.random.triangular(
                        left=0., mode=mode, right=2. * self.mean_gaussian_std, 
                        size=1)[0]
            else:
                raise NotImplementedError()
        else:
            gaussian_noise_std = 0.

        # Brightness (not in original Keras impl.)
        if self.brightness != 0:
            if self.brightness > 1. or self.brightness < 0.:
                raise ValueError('delta must lie within [0, 1]; got %.4f'
                                 % self.brightness)
            if self.rand_distr == 'uniform':
                brightness_delta = np.random.uniform(low=-self.brightness, 
                                                     high=self.brightness)
            elif self.rand_distr == 'normal':
                brightness_delta = truncnorm_cent_2std(
                        min_val=-self.brightness, max_val=self.brightness)
            elif self.rand_distr == 'extremes':
                brightness_delta = np.random.choice([-self.brightness, 
                                                     self.brightness])
            elif self.rand_distr == 'triangular':
                brightness_delta = np.random.triangular(left=-self.brightness, 
                                                        mode=0.,
                                                        right=self.brightness)
            else:
                raise NotImplementedError()
        else:
            brightness_delta = 0.

        # Contrast (not in original Keras impl.)
        if self.contrast[0] != 0 and self.contrast[1] != 0:
            max_gamma_allowed = 2.
            min_gamma_allowed = 0.
            if self.contrast[1] > max_gamma_allowed or \
               self.contrast[0] < min_gamma_allowed:
                raise ValueError( 'gamma range must lie within '
                                  '[%.2f, %.2f]; got [%.4f, %.4f]'
                                  % (min_gamma_allowed, max_gamma_allowed,
                                     self.contrast[0], self.contrast[1]))
            if self.rand_distr == 'uniform':
                contrast_gamma = np.random.uniform(low=self.contrast[0], 
                                                   high=self.contrast[1])
            elif self.rand_distr == 'normal':
                contrast_gamma = truncnorm_cent_2std(min_val=self.contrast[0],
                                                     max_val=self.contrast[1])
            elif self.rand_distr == 'extremes':
                contrast_gamma = np.random.choice([self.contrast[0], 
                                                   self.contrast[1]])
            elif self.rand_distr == 'triangular':
                mode = (self.contrast[0] + self.contrast[1]) / 2.
                contrast_gamma = np.random.triangular(left=self.contrast[0],
                                                      mode=mode,
                                                      right=self.contrast[1])
            else:
                raise NotImplementedError()
        else:
            contrast_gamma = 0.

        # Lab noise (not in original Keras impl.)
        if self.lab_noise_range != 0:
            if self.lab_noise_range < 0:
            # Max absolute value of the random noise. The default 1.3279
            # guarantees that every pixel stays within the perceptual
            # difference range. 
            # See: https://en.wikipedia.org/wiki/Color_difference 
            # If the argument is # negative, the default value is used.
                self.lab_noise_range = 1.3279
            if self.rand_distr == 'uniform':
                lab_px_diff = np.random.uniform(-self.lab_noise_range,
                                                self.lab_noise_range, 
                                                img_shape)
            elif self.rand_distr == 'normal':
                lab_px_diff = truncnorm_cent_2std(-self.lab_noise_range,
                                                  self.lab_noise_range, 
                                                  img_shape)
            elif self.rand_distr == 'extremes':
                lab_px_diff = np.random.choice([-self.lab_noise_range,
                                                self.lab_noise_range], 
                                               img_shape)
            else:
                raise NotImplementedError()
        else:
            lab_px_diff = None

        # Lab training (not in original Keras impl.)
        if self.train_lab:
            x = rgb2lab(x)

        # Image cropping (not in original Keras impl.)
        if self.random_crop and (self.crop_size is not None):
            if np.any(list(img_shape) < self.crop_size):
                raise ValueError('Crop shape cannot be larger than the input')
            limits = img_shape - np.array(self.crop_size) + 1
            if self.rand_distr == 'uniform':
                crop_offsets = np.random.uniform(low=0, high=limits, size=3)
            elif self.rand_distr == 'normal':
                crop_offsets= truncnorm_cent_2std(min_val=0., max_val=limits,
                                             size=3)
            elif self.rand_distr == 'triangular':
                mode = limits / 2.
                crop_offsets = np.random.triangular(left=0, mode=mode, 
                                                    right=limits, size=3)
            elif self.rand_distr == 'extremes':
                crop_offsets = np.asarray([np.random.choice([0, limit - 1])
                    for limit in limits])
            else:
                raise NotImplementedError()
            crop_offsets = crop_offsets.astype(int)
        elif self.central_crop and (self.crop_size is not None):
            if np.any(img_shape < self.crop_size):
                raise ValueError('Crop shape cannot be larger than the input')
            crop_offsets = np.divide(img_shape - np.array(self.crop_size), 2)
        else:
            crop_offsets = None

        # Cutout (not in original Keras impl.)
        if self.cutout[0] != 0 or self.cutout[1] != 0:
            if self.cutout[0] == -1 or self.cutout[1] == -1:
                if self.rand_distr == 'uniform':
                    pct_height, pct_width = np.random.uniform(
                            low=0.25, high=0.75, size=2)
                elif self.rand_distr == 'normal':
                     pct_height, pct_width = truncnorm_cent_2std(
                             min_val=0.25, max_val=0.75, size=2)
                else:
                    raise NotImplementedError()
            else:
                pct_height = self.cutout[0]
                pct_width = self.cutout[1]

            if self.target_shape is not None:
                target_shape = self.target_shape
            else:
                target_shape = img_shape
            cutout_height = pct_height * target_shape[0]
            cutout_width = pct_width * target_shape[1]

            if self.rand_distr == 'uniform':
                cutout_i = np.random.uniform(0, target_shape[0])
                cutout_j = np.random.uniform(0, target_shape[1])
            elif self.rand_distr == 'normal':
                cutout_i = truncnorm_cent_2std(0, target_shape[0])
                cutout_j = truncnorm_cent_2std(0, target_shape[1])
            else:
                raise NotImplementedError()
        else:
            cutout_height = 0
            cutout_width = 0
            cutout_i = 0
            cutout_j = 0

        # Note that the brightness adjustment from the original Keras
        # implementation has been removed completely

        # Transform parameters: new functionality added to dictionary
        transform_parameters = {
            'theta': theta,
            'tx': tx,
            'ty': ty,
            'shear': shear,
            'zx': zx,
            'zy': zy,
            'flip_horizontal': flip_horizontal,
            'flip_vertical': flip_vertical,
            'channel_shift_intensity': channel_shift_intensity,
            'gaussian_noise_std': gaussian_noise_std,
            'brightness_delta': brightness_delta,
            'contrast_gamma': contrast_gamma,
            'lab_px_diff': lab_px_diff,
            'train_lab': self.train_lab,
            'crop_size': self.crop_size,
            'crop_offsets': crop_offsets,
            'cutout_height': cutout_height,
            'cutout_width': cutout_width,
            'cutout_i': cutout_i,
            'cutout_j': cutout_j}

        return transform_parameters
    
    # Overwritten to add new transformations and additional functionality
    def apply_transform(self, x, transform_parameters):
        """
        Applies a transformation to an image according to given parameters.

        Parameters
        ----------
        x : 3D tensor
            A single image to get transformed

        transform_parameters : dict
            The parameter pairs describing the transformation.  Currently, the
            following parameters from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - My new functionality (TODO: add to docstring)

        Returns
        -------
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # Added to original implementation
        if self.color_space == 'rgb':
            x /= 255.
            if self.mean is not None:
                self.mean /= 255.
            if self.std is not None:
                self.std /= 255.

        # Affine transformations: rotation, shear, translation, scaling
        x = apply_affine_transform(x, 
                                   transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=img_row_axis,
                                   col_axis=img_col_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode,
                                   cval=self.cval,
                                   order=self.interpolation_order)

        # Channel shift
        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(
                    x,
                    transform_parameters['channel_shift_intensity'],
                    img_channel_axis)

        # Horizontal flip
        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        # Vertical flip
        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        # Original brightness transformation is removed to perform only my
        # implemenetation

        # Added to original implementation: Gaussian noise, brightness
        # adjustment and contrast adjustment require the images are in the
        # range [0, 1]
        if (self.color_space == 'rgb') & (np.issubdtype(x.dtype, np.integer)):
            scale = np.issubdtype(x.dtype, np.integer)
            x = x.astype(self.dtype)
            x /= float(scale)
        else:
            scale = None

        # Gaussian noise (not in original Keras impl.)
        if transform_parameters.get('gaussian_noise_std', 0) != 0:
            x = gaussian_noise(x, transform_parameters['gaussian_noise_std'],
                               self.color_space)

        # Brightness adjustment (not in original Keras impl.)
        if transform_parameters.get('brightness_delta', 0) != 0:
            x = adjust_brightness(x, transform_parameters['brightness_delta'],
                                  self.color_space)

        # Contrast adjustment (not in original Keras impl.)
        if transform_parameters.get('contrast_gamma', 0) != 0:
            x = adjust_contrast(x, transform_parameters['contrast_gamma'],
                                self.color_space)

        # Image crop (not in original Keras impl.)
        if transform_parameters.get('crop_offsets') is not None:
            x = crop_image(x, transform_parameters['crop_offsets'],
                           transform_parameters['crop_size'])

        # Lab training (not in original Keras impl.)
        if transform_parameters.get('train_lab', False):
            x = rgb2lab(x)
            self.color_space = 'lab'

        # Lab noise (not in original Keras impl.)
        if transform_parameters.get('lab_px_diff') is not None:
            x = lab_noise(x, transform_parameters['lab_px_diff'],
                                      input_space=self.color_space,
                                      output_space=self.color_space)

        # Cutout (not in original Keras impl.)
        if transform_parameters.get('cutout_height', 0) != 0 and \
           transform_parameters.get('cutout_width', 0) != 0:
            x = cutout(x, transform_parameters.get('cutout_height'),
                       transform_parameters.get('cutout_width'),
                       transform_parameters.get('cutout_i'),
                       transform_parameters.get('cutout_j'))

        # Re-scale back to the original range
        if scale:
            x *= scale

        return x

    def random_transform(self, x, seed=None):
        """
        Applies a random transformation to an image.

        Parameters
        ----------
            x: 3D tensor
                Single image.

            seed: int
                Random seed.

        Returns
        -------
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)

