"""
Extended set of tools for data augmentation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import color as skico
from scipy.stats import truncnorm

import keras.backend as K


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


def gaussian_noise(x, std, color_space='rgb'):
    """
    Applies additive zero-centered Gaussian noise

    Parameters
    ----------
    x : ndarray
        RGB image as K.floatx() in the range [0, 1]

    std : float
        Standard deviation of the zero-mean Gaussian noise applied to the image.

    color_space : str
        'rgb' or 'lab'. 'lab' not implemented yet.

    Returns
    -------
    x_noise : ndarray
        Noisy image
    """
    if color_space == 'rgb':
        x_noise = x + np.random.normal(loc=0.0, scale=std, size=x.shape)
        x_noise[x_noise  > 1.] = 1.
        x_noise[x_noise  < 0.] = 0.
    else:
        raise ValueError('color_space must be rgb')

    return x_noise


def crop_image(x, offsets, target_size):
    """
    Crop an image given some offsets and the target size

    Parameters
    ----------
    x : ndarray
        3D image

    offsets : int list
        The offsets for each dimension where the crop begins

    target_size : 1-D array
        Output size of the image. If a dimension should not be cropped, pass 
        the full size of that dimension.

    Returns
    -------
    x_cropped : ndarray
        Cropped image
    """
    x_cropped = x[offsets[0]:offsets[0] + target_size[0],
                  offsets[1]:offsets[1] + target_size[1],
                  offsets[2]:offsets[2] + target_size[2]]

    return x_cropped


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


def rgb2lab(x):
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


def lab2rgb(x):
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


def lab_noise(x, px_noise, input_space='rgb', output_space='lab'):
    """
    Adds uniform noise in the Lab space to an image or batch of images

    Parameters
    ----------
    x : ndarray
        Image or batch of images

    px_noise : array
        Additive noise for each pixel. It must be the shape of the image

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
        x = rgb2lab(x)
    elif input_space == 'lab':
        pass
    else:
        raise NotImplementedError

    x_noisy = x + px_noise

    if output_space == 'rgb':
        x_noisy = lab2rgb(x_noisy)
    elif output_space == 'lab':
        pass
    else:
        raise NotImplementedError

    return x_noisy


def cutout(x, height, width, i, j, square=False):
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

    Returns
    -------
    x_cutout : ndarray
        The transformed image
    """
    # TODO: Write proper docs
    x_shape = x.shape

    if square:
        height = width

    i_top = int(np.max([0, i - height / 2]))
    i_bottom = int(np.min([x_shape[0], i + height / 2]))
    j_left = int(np.max([0, j - width / 2]))
    j_right = int(np.min([x_shape[1], j + width / 2]))

    x_cutout = np.copy(x)
    x_cutout[i_top:i_bottom, j_left:j_right, :] = 0.5

    return x_cutout

