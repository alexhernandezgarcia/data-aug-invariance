"""
Extended set of tools for data augmentation. This builds upon and extends 
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
"""
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from skimage import color as skico
from scipy.stats import truncnorm

import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator, NumpyArrayIterator
from keras.preprocessing.image import apply_transform
from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import random_channel_shift, flip_axis
from keras.preprocessing.image import array_to_img

import warnings
import os


def truncnorm_cent_2std(min_val, max_val, size=None):
    """
    Generates n random samples from a truncated normal distribution within the
    range (min_val, max_val), with mean centered within the interval and 
    standard deviation such that the interval covers 2 sigmas (95 %) of the
    normal distribution.

    Parameters
    ----------
    min_val : float
        Minimum value that the samples can take

    max_val : float
        Maximum value that the samples can take

    n : int or tuple of ints
        Number of samples to generate, output shape

    Returns
    -------
    distr : ndarray
        The random samples
    """
    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)

    loc = (min_val + max_val) / 2.
    scale = (max_val - min_val) / 4.

    if size:
        samples = np.zeros(size)
	b = np.broadcast(loc, scale, min_val, max_val, samples)
        idx = 0
	for n_dim, (loc_v, scale_v, min_val_v, max_val_v, _) in enumerate(b):
            while True:
                s = np.random.normal(loc=loc_v, scale=scale_v)
                if (s >= min_val_v) & (s <= max_val_v):
	            samples[np.unravel_index(idx, size)] = s
                    idx += 1
                    break
        return samples
    else:
        while True:
            s = np.random.normal(loc=loc, scale=scale)
            if (s >= min_val) & (s <= max_val):
                return s


def truncnorm_cent_2std_scipy(min_val, max_val, size=None):
    """
    Generates n random samples from a truncated normal distribution within the
    range (min_val, max_val), with mean centered within the interval and 
    standard deviation such that the interval covers 2 sigmas (95 %) of the
    normal distribution.

    Parameters
    ----------
    min_val : float
        Minimum value that the samples can take

    max_val : float
        Maximum value that the samples can take

    n : int or tuple of ints
        Number of samples to generate, output shape

    Returns
    -------
    distr : ndarray
        The random samples
    """
    loc = (min_val + max_val) / 2.
    scale = (max_val - min_val) / 4.
    a = (min_val - loc) / scale
    b = (max_val - loc) / scale

    return truncnorm.rvs(a, b, loc, scale, size)


def gaussian_noise(x, std, color_space='rgb', rand_distr='normal'):
    """
    Applies additive zero-centered Gaussian noise

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1]

    std : float
        Standard deviation of the Gaussian noise

    color_space : str
        'rgb' or 'lab'. 'lab' not implemented yet.

    Returns
    -------
    x_noise : ndarray
        Noisy image
    """
    if rand_distr == 'uniform':
        std = np.random.uniform(low=0., high=2 * std)
    elif rand_distr == 'normal':
        std = truncnorm_cent_2std(min_val=0., max_val=2 * std)
    elif rand_distr == 'triangular':
        mode = (0. + 2 * std) / 2.
        std = np.random.triangular(left=0., mode=mode, right=2 * std, size=1)[0]
    elif rand_distr == 'extremes':
        std = 2 * std
    else:
        raise NotImplementedError('Possible distributions are uniform, normal, '
                                  'triangular and extremes')
        
    std = truncnorm_cent_2std(min_val=0., max_val=2 * std)
    if color_space == 'rgb':
        x_noise = x + np.random.normal(loc=0.0, scale=std, size=x.shape)
        x_noise[x_noise  > 1.] = 1.
        x_noise[x_noise  < 0.] = 0.
    else:
        raise ValueError('color_space must be rgb')

    return x_noise


def random_crop(x, size, rand_distr='normal'):
    """
    Randomly crops an image to a given size

    Parameters
    ----------
    x : ndarray
        RGB image

    size : 1-D array
        Output size of the image. If a dimension should not be cropped, pass 
        the full size of that dimension.

    rand_distr : 'str'
        Distribution from which to sample the random delta. Either 'normal',
        'uniform' or 'triangular'.

    Returns
    -------
    x_cropped : ndarray
        Cropped image
    """
    x_shape = x.shape
    if np.any(x_shape < size):
        raise ValueError('Output crop shape cannot be larger than the input')

    limits = x_shape - np.array(size) + 1

    if rand_distr == 'uniform' or rand_distr == 'extremes':
        offsets = np.random.uniform(low=0, high=limits, size=3).astype(int)
    elif rand_distr == 'normal':
        offsets = truncnorm_cent_2std(min_val=0., max_val=limits,
                                             size=3).astype(int)
    elif rand_distr == 'triangular':
        mode = limits / 2.
        offsets = np.random.triangular(left=0, mode=mode, right=limits,
                                  size=3).astype(int)
    else:
        raise NotImplementedError('Possible distributions are uniform, normal, '
                                  'triangular and extremes')

    x_cropped = x[offsets[0]:offsets[0] + size[0],
                  offsets[1]:offsets[1] + size[1],
                  offsets[2]:offsets[2] + size[2]]

    return x_cropped


def central_crop(x, size):
    """
    Crops an image at is center to a given size

    Parameters
    ----------
    x : ndarray
        RGB image

    size : 1-D array
        Output size of the image. If a dimension should not be cropped, pass 
        the full size of that dimension.

    Returns
    -------
    x_cropped : ndarray
        Cropped image
    """
    x_shape = x.shape
    if np.any(x_shape < size):
        raise ValueError('Output crop shape cannot be larger than the input')

    offsets = np.divide(x_shape - np.array(size), 2)
    x_cropped = x[offsets[0]:offsets[0] + size[0],
                  offsets[1]:offsets[1] + size[1],
                  offsets[2]:offsets[2] + size[2]]

    return x_cropped


def random_brightness(x, max_delta, color_space='rgb', rand_distr='normal'):
    """
    Adjust the brightness of an image by a uniformly random delta in the range 
    [-max_delta, max_delta]

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1] OR
        Lab image as K.floatx() with L channel in the range [0, 100]

    max_delta : float
        Maximum absolute value (it must be non-negative) to define the range of
        the random delta.

    color_space : str
        'rgb' or 'lab'

    rand_distr : 'str'
        Distribution from which to sample the random delta. Either 'normal',
        'uniform' or 'triangular'.

    Returns
    -------
    x_adjusted : ndarray
        Image randomly adjusted by brightness
    """
    if (max_delta > 1.) | (max_delta < 0.):
        raise ValueError('max delta must lie within [0, 1]; got %.4f'
                % max_delta)

    if rand_distr == 'uniform':
        delta = np.random.uniform(low=-max_delta, high=max_delta, size=1)[0]
    elif rand_distr == 'normal':
        delta = truncnorm_cent_2std(min_val=-max_delta, max_val=max_delta)
    elif rand_distr == 'triangular':
        delta = np.random.triangular(left=-max_delta, mode=0., right=max_delta,
                                  size=1)[0]
    elif rand_distr == 'extremes':
        delta = np.random.choice([-max_delta, max_delta])
    else:
        raise NotImplementedError('Possible distributions are uniform, normal, '
                                  'triangular and extremes')

    x_adjusted = adjust_brightness(x, delta, color_space)

    return x_adjusted


def adjust_brightness(x, delta, color_space='rgb'):
    """
    Adjust the brightness of an image by a delta.

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1] OR
        Lab image as K.floatx() with L channel in the range [0, 100]

    delta : float
        Amount to add to increase/decrease the brightness of the image. It 
        should be in the range [-1, 1]

    color_space : str
        'rgb' or 'lab'

    Returns
    -------
    x : ndarray
        Image adjusted by brightness
    """

    if (delta > 1.) | (delta < -1.):
        raise ValueError('delta must lie within [-1, 1]; got %.4f' % delta)

    if color_space == 'rgb':
        x += delta
        x[x > 1.] = 1.
        x[x < 0.] = 0.
    elif color_space == 'lab':
        l_ch = x[:, :, 0]
        l_ch += delta * 100.
        l_ch[l_ch > 100.] = 100.
        l_ch[l_ch < 0.] = 0.
    else:
        raise ValueError('color_space must be either rgb or lab')

    return x


def random_contrast(x, min_gamma, max_gamma, color_space='rgb', 
                    rand_distr='normal'):
    """
    Adjust the contrast of an image by a uniformly random gamma in the range 
    [-min_gamma, max_gamma]

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1] OR
        Lab image as K.floatx() with L channel in the range [0, 100]

    min_gamma : float
        Minimum value to define the range of the uniform random gamma.

    max_gamma : float
        Minimum value to define the range of the uniform random gamma.

    color_space : str
        'rgb' or 'lab'

    rand_distr : 'str'
        Distribution from which to sample the random delta. Either 'normal',
        'uniform' or 'triangular'.

    Returns
    -------
    x_adjusted : ndarray
        Image randomly adjusted by contrast
    """
    max_gamma_allowed = 2.
    min_gamma_allowed = 0.
    if (max_gamma > max_gamma_allowed) | (min_gamma < min_gamma_allowed):
        raise ValueError(
                'gamma range must lie within [%.2f, %.2f]; got [%.4f, %.4f]'
                % (min_gamma_allowed, max_gamma_allowed, min_gamma, max_gamma))

    if rand_distr == 'uniform':
        gamma = np.random.uniform(low=min_gamma, high=max_gamma, size=1)[0]
    elif rand_distr == 'normal':
        gamma = truncnorm_cent_2std(min_val=min_gamma, max_val=max_gamma)
    elif rand_distr == 'triangular':
        mode = (min_gamma + max_gamma) / 2.
        gamma = np.random.triangular(left=min_gamma, mode=mode,
                                     right=max_gamma, size=1)[0]
    elif rand_distr == 'extremes':
        gamma = np.random.choice([min_gamma, max_gamma])
    else:
        raise NotImplementedError('Possible distributions are uniform, normal, '
                                  'triangular and extremes')

    x_adjusted = adjust_contrast(x, gamma, color_space)

    return x_adjusted


def adjust_contrast(x, gamma, color_space='rgb'):
    """
    Adjust the contrast of an image according to the formula: 
    x = (x - mean(x)) * gamma + mean(x).

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1] OR
        Lab image as K.floatx() with L channel in the range [0, 100]

    gamma : float
        Factor by which the contrast is adjusted. It should be a non-negative 
        value. If gamma is larger than 1, the contrast is increased; if gamma 
        is smaller than 1, the contrast is reduced.

    color_space : str
        'rgb' or 'lab'

    Returns
    -------
    x : ndarray
        Image adjusted by brightness
    """

    if gamma < 0:
        raise ValueError('gamma must be non-negative; got %.4f' % gamma)

    if color_space == 'rgb':
        x_mean = np.mean(x, axis=(0, 1), keepdims=True)     
        x = (x - x_mean) * gamma + x_mean
        x[x > 1.] = 1.
        x[x < 0.] = 0.
    elif color_space == 'lab':
        l_ch = x[:, :, 0]
        l_mean = np.mean(l_ch, keepdims=True)
        l_ch = (l_ch - l_mean) * gamma + l_mean
        l_ch[l_ch > 100.] = 100.
        l_ch[l_ch < 0.] = 0.
        x[:, :, 0] = l_ch
    else:
        raise ValueError('color_space must be either rgb or lab')

    return x


def adjust_saturation(x, gamma, color_space='rgb'):
    """
    Adjust the contrast of an image according to the formula: 
    x = (x - mean(x)) * gamma + mean(x).

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1] OR
        Lab image as K.floatx() with L channel in the range [0, 100]

    gamma : float
        Factor by which the contrast is adjusted. It should be a non-negative 
        value. If gamma is larger than 1, the contrast is increased; if gamma 
        is smaller than 1, the contrast is reduced.

    color_space : str
        'rgb' or 'lab'

    Returns
    -------
    x : ndarray
        Image adjusted by brightness
    """

    if gamma < 0:
        raise ValueError('gamma must be non-negative')

    if color_space == 'rgb':
        x_mean = np.mean(x, axis=(0, 1), keepdims=True)     
        x = (x - x_mean) * gamma + x_mean
        x[x > 1.] = 1.
        x[x < 0.] = 0.
    elif color_space == 'lab':
        l_ch = x[:, :, 0]
        l_mean = np.mean(l_ch, keepdims=True)
        l_ch = (l_ch - l_mean) * gamma + l_mean
        l_ch[l_ch > 100.] = 100.
        l_ch[l_ch < 0.] = 0.
        x[:, :, 0] = l_ch
    else:
        raise ValueError('color_space must be either rgb or lab')

    return x


def rgb_to_lab(x):
    """
    Converts an RGB image into the CIE Lab space

    Parameters
    ----------
    x : ndarray
        RGB image

    Returns
    -------
    x_lab : ndarray
        Lab image
    """
    # Convert the input image into float32
    if x.dtype == 'uint8':
        x = x.astype(K.floatx())
        x /= 255.
    else:
        if np.max(x) > 1.:
            x /= 255.
    x_lab = skico.rgb2lab(x)

    return x_lab


def lab_to_rgb(x):
    """
    Converts a CIE Lab image into RGB, dtype=K.floatx() in [0, 1]

    Parameters
    ----------
    x : ndarray
        Lab image

    Returns
    -------
    x_rgb : ndarray
        RGB image
    """
    x_rgb = skico.lab2rgb(x)
    x_rgb = x_rgb.astype(K.floatx())

    return x_rgb


def add_lab_uniform_noise(x, max_pix_diff=1.3279, input_space='rgb', 
                          output_space='lab'):
    """
    Adds uniform noise in the Lab space to an image or batch of images

    Parameters
    ----------
    x : ndarray
        Image or batch of images

    max_pix_diff : float
        Max absolute value of the random noise. The default 1.3279 guarantees 
        that every pixel stays within the perceptual difference range. See 
        https://en.wikipedia.org/wiki/Color_difference
        If the argument is negative, the default value is used.

    input_space : str
        Color space of the input x. Default: 'rgb'. Other options: 'lab'

    output_space : str
        Color space of the output image. Default: 'lab'. Other options: 'rgb'

    Returns
    -------
    x_noisy : ndarray
        Noisy image
    """
    if input_space == 'rgb':
        x = rgb_to_lab(x)
    elif input_space == 'lab':
        pass
    else:
        raise NotImplementedError

    if max_pix_diff < 0:
        max_pix_diff = 1.3279
    u_noise = np.random.uniform(-max_pix_diff, max_pix_diff, x.shape)
    x_noisy = x + u_noise

    if output_space == 'rgb':
        x_noisy = lab_to_rgb(x_noisy)
    elif output_space == 'lab':
        pass
    else:
        raise NotImplementedError

    return x_noisy


def cutout(x, pct_width=-1, pct_height=-1, min_pct=0.25, max_pct=0.75,
           square=False, color_space='rgb', rand_distr_size='uniform',
           rand_distr_loc='uniform'):
    """
    Performs "cutout" augmentation on an image, that is masking the image with
    a rectangular gray area.

    Parameters
    ----------
    x : ndarray
        Image or batch of images

    pct_width : float
        The percentage of the image width that the mask takes. If -1, it is
        randomly sampled.

    pct_height : float
        The percentage of the image height that the mask takes. If -1, it is
        randomly sampled.

    min_pct : float
        The minimum percentage of image width/height to use for randomly
        sampling the percentages.

    max_pct : float
        The maximum percentage of image width/height to use for randomly
        sampling the percentages.

    square : bool
        If True, the mask shape is forcedly a square

    color_space : str
        'rgb' or 'lab'

    rand_distr_size : 'str'
        Distribution from which to sample the random mask size.
        Either 'normal' or 'uniform'.

    rand_distr_loc : 'str'
        Distribution from which to sample the random location of the mask.
        Either 'normal' or 'uniform'.

    Returns
    -------
    x_cutout : ndarray
        The transformed image
    """
    x_shape = x.shape

    if pct_width == -1 or pct_height == -1:
        if rand_distr_size == 'uniform':
            pct_width, pct_height = np.random.uniform(
                    low=min_pct, high=max_pct, size=2)
        elif rand_distr_size == 'normal':
             pct_width, pct_height = truncnorm_cent_2std(
                     min_val=min_pct, max_val=max_pct, size=2)

    height = int(pct_height * x_shape[0])
    width = int(pct_width * x_shape[1])

    if square:
        pct_height = pct_width

    if rand_distr_loc == 'uniform':
        i = np.random.uniform(low=0, high=x_shape[0], size=1).astype(int)[0]
        j = np.random.uniform(low=0, high=x_shape[1], size=1).astype(int)[0]
    elif rand_distr_loc == 'normal':
        i = truncnorm_cent_2std(min_val=0, max_val=x_shape[0],
                                size=1).astype(int)[0]
        j = truncnorm_cent_2std(min_val=0, max_val=x_shape[1],
                                size=1).astype(int)[0]

    i_top = np.max([0, i - height / 2])
    i_bottom = np.min([x_shape[0], i + height / 2])
    j_left = np.max([0, j - width / 2])
    j_right = np.min([x_shape[1], j + width / 2])

    x_cutout = np.copy(x)
    x_cutout[i_top:i_bottom, j_left:j_right, :] = 0.5

    return x_cutout


class ImageDataGeneratorExt(ImageDataGenerator):

    def __init__(self,
                 color_space='rgb',
                 gaussian_std=0.,
                 do_random_crop=False,
                 do_central_crop=False,
                 crop_size=None,
                 brightness=0.,
                 contrast=[0., 0.],
                 cutout=[0., 0.],
                 lab_u_noise=0.,
                 train_lab=False,
                 rand_distr='normal',
                 seed_daug=None,
                 *args,
                 **kwargs):
        super(ImageDataGeneratorExt, self).__init__(*args, **kwargs)
        self.color_space = color_space
        self.gaussian_std = gaussian_std
        self.random_crop = do_random_crop
        self.central_crop = do_central_crop
        self.crop_size = crop_size
        self.brightness = brightness
        self.contrast = contrast
        self.cutout = cutout
        self.lab_u_noise = lab_u_noise
        self.train_lab = train_lab
        self.rand_distr = rand_distr
        self.seed_daug=seed_daug

        if (self.random_crop | self.central_crop) & \
           (self.crop_size is not None):
            self.im_shape = self.crop_size
        else:
            self.im_shape = None

    # This function is overwritten in order to allow for image cropping and
    # training with image identification
    def flow(self, x, y=None, batch_size=32, aug_per_im=1, shuffle=True, 
             seed_shuffle=None, save_to_dir=None, save_prefix='', 
             save_format='jpeg'):

        # Added to original implementation to support image cropping
        if self.im_shape is None:
            im_shape = list(x.shape)[1:]
        else:
            im_shape = self.im_shape

        # Added to original implementation to support list of labels y
        if isinstance(y, list):
            y_list = [np.asarray(e) for e in y]
            y = y[0]
        else:
            y_list = None

        return DaskArrayIterator(
            x, y, im_shape, y_list, self, 
            batch_size=batch_size,
            aug_per_im=aug_per_im,
            shuffle=shuffle,
            seed=seed_shuffle,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def random_transform(self, x):
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if self.seed_daug is not None:
            np.random.seed(self.seed_daug)
            if self.seed_daug < np.iinfo(type(self.seed_daug)).max:
                self.seed_daug +=1
            else:
                self.seed_daug = 1

        # Added to original implementation
        if self.color_space == 'rgb':
            x /= 255.
            if self.mean is not None:
                self.mean /= 255.
            if self.std is not None:
                self.std /= 255.

        # Use composition of homographies to generate final transform that 
        # needs to be applied
        if self.rotation_range:
            if self.rand_distr == 'uniform':
                theta = np.pi / 180 * np.random.uniform(-self.rotation_range, 
                                                        self.rotation_range)
            elif self.rand_distr == 'normal':
                theta = np.pi / 180 * truncnorm_cent_2std(-self.rotation_range, 
                                                          self.rotation_range)
            elif self.rand_distr == 'extremes':
                theta = np.pi / 180 * np.random.choice([-self.rotation_range,
                                                        self.rotation_range])
            else:
                raise NotImplementedError()
        else:
            theta = 0

        if self.height_shift_range:
            if self.rand_distr == 'uniform':
                tx = np.random.uniform(-self.height_shift_range, 
                                       self.height_shift_range) * \
                     x.shape[img_row_axis]
            elif self.rand_distr == 'normal':
                tx = truncnorm_cent_2std(-self.height_shift_range, 
                                         self.height_shift_range) * \
                     x.shape[img_row_axis]
            elif self.rand_distr == 'extremes':
                tx = np.random.choice([-self.height_shift_range,
                                       self.height_shift_range]) * \
                     x.shape[img_row_axis]
            else:
                raise NotImplementedError()
        else:
            tx = 0

        if self.width_shift_range:
            if self.rand_distr == 'uniform':
                ty = np.random.uniform(-self.width_shift_range, 
                                       self.width_shift_range) * \
                                       x.shape[img_col_axis]
            elif self.rand_distr == 'normal':
                ty = truncnorm_cent_2std(-self.width_shift_range, 
                                         self.width_shift_range) * \
                                         x.shape[img_col_axis]
            elif self.rand_distr == 'extremes':
                ty = np.random.choice([-self.width_shift_range,
                                       self.width_shift_range]) * \
                     x.shape[img_col_axis]
            else:
                raise NotImplementedError()
        else:
            ty = 0

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

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix 
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix,
                                                              h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        # Added to original implementation
        if self.gaussian_std != 0:
            x = gaussian_noise(x, self.gaussian_std, self.color_space,
                               self.rand_distr)

        # Added to original implementation
        if self.brightness != 0:
            x = random_brightness(x, self.brightness, self.color_space,
                                  self.rand_distr)

        # Added to original implementation
        if self.contrast[0] != 0 and self.contrast[1] != 0:
            x = random_contrast(x, self.contrast[0], self.contrast[1], 
                                self.color_space, self.rand_distr)

        # Added to original implementation
        if self.cutout[0] != 0 or self.cutout[1] != 0:
            x = cutout(x, self.cutout[0], self.cutout[1],
                       color_space=self.color_space,
                       rand_distr_size=self.rand_distr,
                       rand_distr_loc=self.rand_distr)

        # Added to original implementation
        if self.lab_u_noise != 0:
            x = add_lab_uniform_noise(x, max_pix_diff=self.lab_u_noise, 
                                      output_space='rgb')

        # Added to original implementation
        if self.train_lab:
            x = rgb_to_lab(x)

        # Added to original implementation to support image cropping
        if self.random_crop & (self.crop_size is not None):
            x = random_crop(x, self.crop_size, self.rand_distr)
        elif self.central_crop & (self.crop_size is not None):
            x = central_crop(x, self.crop_size)

        return x


class NumpyArrayIteratorExt(NumpyArrayIterator):
    """We extend NumpyArrayInterator in order to allow for image cropping and a
    list of labels y.

    In the original code, in _get_batches_of_transformed_samples() batch_x is 
    initialized using x.shape, which is the original image shape. However, if 
    there is cropping x.shape is changed within random_transform(). We sort 
    this out by adding an extra attribute self.im_shape and using it in the 
    initialization of batch_x. The additional y_list argument allows for 
    multiple outputs.
    """

    def __init__(self,
                 im_shape,
                 y_list,
                 *args,
                 **kwargs):
        super(NumpyArrayIteratorExt, self).__init__(*args, **kwargs)
        self.im_shape = im_shape
        self.y_list = y_list

    # Overwritten to support image cropping and a list of labels y
    def _get_batches_of_transformed_samples(self, index_array):
        # IMPORTANT: the next line is changed with respect to the original 
        # keras implementation Change: self.im_shape <-- list(self.x.shape)[1:]
        batch_x = np.zeros(tuple([len(index_array)] + self.im_shape), 
                           dtype=K.floatx())
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(
                    x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                            prefix=self.save_prefix,
                            index=j,
                            hash=np.random.randint(1e4),
                            format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x

        # Added to original implementation to support lists of labels y
        if self.y_list:
            batch_y = [y[index_array] for y in self.y_list]
        else:
            batch_y = self.y[index_array]

        return batch_x, batch_y


class DaskArrayIterator(Iterator):
    """
    Iterator yielding data from a dask array.

    Arguments
    ---------
    x : dask.array
        dask array of input data

    y : dask.array
        dask array of targets data

    im_shape : list
        Shape of the images. Necessary if cropping is performed. 

    y_list : list
        List of y labels. Necessary for networks with multiple outputs.

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
    
    def __init__(self, x, y, im_shape, y_list, image_data_generator,
                 batch_size=32, aug_per_im=1, shuffle=False, seed=None, 
                 data_format=None, save_to_dir=None, save_prefix='',
                 save_format='png'):
        # Note that most lines are copied from the __init__ function of
        # NumpyArrayIterator. Importantly, np.asarray(x) (or on y) is never 
        # performed here, since the memory could be filled up.
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (x.shape, y.shape))

        if data_format is None:
            data_format = K.image_data_format()
        
        self.x_dask = x

        # Define the chunk size. The size is assumed to be the 0th dimension of
        # the chunks and the other dimensions should be equal to x.shape[1:]
        self.chunk_size = self.x_dask.chunks[0][0]

        self.x = np.asarray(self.x_dask[:self.chunk_size], dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                          '(channels on axis ' + str(channels_axis) + '), '
                          'i.e. expected either 1, 3 or 4 channels on axis ' 
                          + str(channels_axis) + '. However, it was passed an '
                          'array with shape ' + str(self.x.shape) + ' (' + 
                          str(self.x.shape[channels_axis]) + ' channels).')

        if y is not None:
            self.y_dask = y
            self.y = np.asarray(self.y_dask[:self.chunk_size])
        else:
            self.y = None
            self.y_dask = y

        self.im_shape = im_shape
        self.y_list = y_list
        self.image_data_generator = image_data_generator
        self.n_images = self.x_dask.shape[0]
        self.n_aug = aug_per_im
        self.chunk_index = 0
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        super(DaskArrayIterator, self).__init__(self.chunk_size, batch_size, 
                                                shuffle, seed)

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
                                dtype=K.floatx())
            self.y = np.asarray(self.y_dask[current_chunk_index:
                                            current_chunk_index + self.n])
            self.n = self.x.shape[0]
            self._set_index_array() 
            
        # Check last chunk
        if idx == len(self) - 1:
            self.chunk_index += 1

        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]

        return self._get_batches_of_transformed_samples(index_array)

    # Overwritten to support image cropping and a list of labels y
    def _get_batches_of_transformed_samples(self, index_array):
        # IMPORTANT: the next line is changed with respect to the original 
        # keras implementation Change: self.im_shape <-- list(self.x.shape)[1:]
        # Additionally: support for more than one augmentation per image
        batch_x = np.zeros(tuple([len(index_array) * self.n_aug] \
                                 + self.im_shape), dtype=K.floatx())
        batch_y_id = np.zeros([len(index_array) * self.n_aug] * 2,
                              dtype=np.uint8)
        for i, j in enumerate(index_array):
            batch_y_id[i * self.n_aug:i * self.n_aug + self.n_aug,
                       i * self.n_aug:i * self.n_aug + self.n_aug] = 1
            for k in range(self.n_aug):
                x = self.x[j]
                x = self.image_data_generator.random_transform(
                        x.astype(K.floatx()))
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
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            idx = np.random.permutation(batch_x.shape[0])
        else:
            idx = np.arange(batch_x.shape[0])

        batch_x = batch_x[idx]
        batch_y_id = batch_y_id[idx][:, idx]

        if self.y is None:
            return batch_x

        # Added to original implementation to support lists of labels y and 
        # more than one augmentation per image
        if self.y_list:
            batch_y = [np.repeat(y[index_array], self.n_aug, axis=0) for y 
                       in self.y_list]
            batch_y = [y[idx] for y in batch_y]
        else:
            batch_y = np.repeat(self.y[index_array], self.n_aug, axis=0)
            batch_y = batch_y[idx]
        batch_y_list = [batch_y, batch_y_id]

        return batch_x, batch_y_list

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

    def reset(self):
        self.batch_index = 0
        self.chunk_index = 0
        self.x = np.asarray(self.x_dask[:self.chunk_size], dtype=K.floatx())
        if self.y is not None:
            self.y = np.asarray(self.y_dask[:self.chunk_size])
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
                                    dtype=K.floatx())
                self.y = np.asarray(self.y_dask[current_chunk_index:
                                                current_chunk_index + self.n])
                self.n = self.x.shape[0]
                self._set_index_array()
                
            current_batch_index = (self.batch_index * self.batch_size) % self.n
            self.batch_index += 1
            self.total_batches_seen += 1
                
            yield self.index_array[current_batch_index:
                                   current_batch_index + self.batch_size]

