import math
import sys
from collections import namedtuple

import numpy as np
from numpy.random import random


ValueRange = namedtuple('ValueRange', 'min max')


def get_random(min_value, max_value):
    return random() * max_value + min_value


def clamp(value, min_value, max_value):
    return min_value if value < min_value else max_value if max_value < value else value


def get_squared_error(feature_vector_a, feature_vector_b):
    ''' returns squared euclidean distance '''

    error_list = np.subtract(feature_vector_a, feature_vector_b)
    squared_error = np.sum(
        np.multiply(error_list, error_list)
    )

    return squared_error


def get_euclidean_similarity(feature_vector_a, feature_vector_b):
    ''' returns euclidean similarity '''
    squared_error = get_squared_error(feature_vector_a, feature_vector_b)
    distance = math.sqrt(squared_error)

    return 1.0 / (1.0 + distance)


def inner_prod(feature_vector_a, feature_vector_b):
    prod = np.sum(
        np.multiply(feature_vector_a.weights, feature_vector_b.weights)
    )

    return prod


def get_squared_distance(ax, ay, bx, by):
    x_delta = ax - bx
    y_delta = ay - by
    sqr_dist = x_delta * x_delta + y_delta * y_delta

    return sqr_dist


def get_euclidean_distance(self, point2d):
    sqr_dist = self.get_squared_distance(point2d)

    return math.sqrt(sqr_dist)


def get_cosine_similarity(feature_vector_a, feature_vector_b):
    ''' returns cosine similarity '''

    prod = inner_prod(feature_vector_a, feature_vector_b)

    norm_a = feature_vector_a.get_norm()
    norm_b = feature_vector_b.get_norm()

    if norm_a == 0 or norm_b == 0:
        similarity = float('inf')
    else:
        similarity = prod / norm_a * norm_b

    return similarity


def get_norm(feature_vector):
    ''' returns feature_vector's norm value '''
    norm = math.sqrt(
        np.sum(
            np.multiply(feature_vector.weights, feature_vector.weights)
        )
    )

    return norm


def print_map(som, sample_map, idx):
    sample = sample_map.map[idx]

    sample_error_map = np.subtract(som.map, sample)
    sample_error_map = np.multiply(sample_error_map, sample_error_map)
    sample_error_map = np.sum(sample_error_map, axis=2)

    sample_max_err = np.max(sample_error_map)
    sample_a_sim_map = np.divide(sample_error_map, sample_max_err)
    bmu_idx_list = [som.get_bmu_coord(unit) for unit in sample_map.map]

    print('--' * som.width)
    for x in range(som.width):
        for y in range(som.height):
            i = sample_a_sim_map[x][y]

            try:
                mark = bmu_idx_list.index([x, y])
                mark = f'{str(mark)[:2]:>2}'
            except Exception:
                mark = '  '

            ansi_color_idx = 232 + int(i * 256 / 24)
            sys.stdout.write(
                f"\033[48;5;{ansi_color_idx}m{mark}\033[0m")
        sys.stdout.write('\n')
    sys.stdout.write('\n')
