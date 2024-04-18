#!/usr/bin/env python
# Created by "Thieu" at 02:44, 04/12/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from numbers import Number


def get_correct_shape(shape):
    if isinstance(shape, Number):
        return (shape, 1)
    elif type(shape) in (list, tuple, np.ndarray) and len(shape) == 1:
        return (shape[0], 1)
    else:
        return shape


def get_generator(seed=None):
    return np.random.default_rng(seed)


def orthogonal_initializer(shape, gain=1.0, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = generator.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def he_uniform_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    limit = np.sqrt(6 / fan_in)
    return generator.uniform(-limit, limit, shape)


def he_normal_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    stddev = np.sqrt(2 / fan_in)
    return generator.normal(0.0, stddev, shape)


def glorot_uniform_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in, fan_out = shape
    limit = np.sqrt(6 / (fan_in + fan_out))
    return generator.uniform(-limit, limit, shape)


def glorot_normal_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in, fan_out = shape
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return generator.normal(0.0, stddev, shape)


def lecun_uniform_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    limit = np.sqrt(3 / fan_in)
    return generator.uniform(-limit, limit, shape)


def lecun_normal_initializer(shape, seed=None):
    generator = get_generator(seed)
    shape = get_correct_shape(shape)
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    stddev = np.sqrt(1 / fan_in)
    return generator.normal(0.0, stddev, shape)


def random_uniform_initializer(shape, minval=0.0, maxval=1.0, seed=None):
    generator = get_generator(seed)
    return generator.uniform(minval, maxval, shape)


def random_normal_initializer(shape, mean=0.0, stddev=1.0, seed=None):
    generator = get_generator(seed)
    return generator.normal(mean, stddev, shape)
