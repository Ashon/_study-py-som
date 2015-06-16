
import math
from collections import namedtuple
from numpy.random import random


ValueRange = namedtuple('ValueRange', 'min max')

def get_random(min_value, max_value):
    return random() * max_value + min_value

def clamp(value, min_value, max_value):
    return min_value if value < min_value else max_value if max_value < value else value

def get_squared_error(feature_vector_a, feature_vector_b):
    ''' returns squared euclidean distance '''

    weight_length = feature_vector_a.get_dimension()
    error_list = [
        feature_vector_a.weights[i] - feature_vector_b.weights[i]
        for i in range(weight_length)
    ]
    squared_error = sum([error * error for error in error_list])

    return float(squared_error)

def get_euclidean_similarity(feature_vector_a, feature_vector_b):
    ''' returns euclidean similarity '''
    squared_error = feature_vector_a.get_squared_error(feature_vector_b)
    distance = math.sqrt(squared_error)

    return 1.0 / (1.0 + distance)

def inner_prod(feature_vector_a, feature_vector_b):
    dimension = feature_vector_a.get_dimension()

    prod = sum([
        feature_vector_a.weights[i] * feature_vector_b.weights[i]
        for i in range(dimension)
    ])

    return prod

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
    norm = math.sqrt(sum([weight * weight for weight in feature_vector.weights]))

    return norm
