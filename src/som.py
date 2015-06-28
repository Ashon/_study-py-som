
'''
    self organizing map

    @author ashon
'''

import time
import numpy as np
import som_util

def clamp(value, min_value, max_value):
    return min_value if value < min_value else \
        max_value if max_value < value else value

class FeatureMap(object):

    def __init__(self, width=0, height=0, dimension=0, randomize=False):

        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        self.map = fill_method(
            width * height * dimension
        ).reshape(width, height, dimension)

        self._width = width
        self._width_range = range(self._width)

        self._height = height
        self._height_range = range(self._height)
        self._dimension = dimension

        som_util.log_with_args(
            log_level='INFO', instance=self, dimension=self._dimension,
            width=self._width, height=self._height)

    def get_error_map(self, feature_vector):

        start_time = time.time()

        error_map = np.add(-self.map, feature_vector)

        exec_time = time.time() - start_time
        som_util.log_with_args(exec_time=exec_time, message='get error map')

        return error_map

    def get_bmu_coord(self, error_map):
        ''' Returns best matching unit's coord.
            Select nearist neighbor.
        '''

        start_time = time.time()

        squared_error_map = np.multiply(error_map, error_map)
        sum_squared_error_map = np.sum(squared_error_map, axis=2)
        min_error = np.amin(sum_squared_error_map)
        min_error_coords = np.where(sum_squared_error_map == min_error)

        bmu_coord = np.array([
            min_error_coords[0][0],
            min_error_coords[1][0]
        ])

        exec_time = time.time() - start_time
        som_util.log_with_args(
            exec_time=exec_time, message='found bmu',
            x=bmu_coord[0], y=bmu_coord[1])

        return bmu_coord

    def get_bmu(self, error_map):
        ''' returns bmu '''
        bmu_coord = self.get_bmu_coord(error_map)

        return self.map[bmu_coord]


class Som(FeatureMap):

    def __init__(self, width=0, height=0, dimension=0, randomize=False,
                 threshold=0.5, learning_rate=0.05, max_iteration=5, gain=2):

        super(Som, self).__init__(
            width=width, height=height,
            dimension=dimension, randomize=randomize)

        self._threshold = clamp(threshold, 0, 1)
        self._learning_rate = clamp(learning_rate, 0, 1)
        self._learn_threshold = self._threshold * self._learning_rate

        self._gain = gain
        self._iteration = 0
        self._max_iteration_count = int(max_iteration)

        som_util.log_with_args(
            log_level='INFO', instance=self, threshold=self._threshold, gain=self._gain,
            learning_rate=self._learning_rate, iteration=self._max_iteration_count
        )

    def _set_learn_threshold(self):
        self._learn_threshold = self._threshold * self._learning_rate

    def set_learning_rate(self, learning_rate):
        self._learning_rate = clamp(learning_rate, 0, 1)
        self._set_learn_threshold()

    def get_learning_rate(self):
        return self._learning_rate

    def set_threshold(self, threshold):
        self._threshold = clamp(threshold, 0, 1)
        self._set_learn_threshold()

    def get_threshold(self):
        return self._threshold

    def get_learn_threshold(self):
        return self._learn_threshold

    def set_iteration_count(self, iteration_count):
        self._max_iteration_count = iteration_count

    def get_iteration_count(self):
        return self._max_iteration_count

    def do_progress(self):
        self._iteration += 1

    def get_progress(self):
        return float(self._iteration) / self._max_iteration_count

    def get_activation_map(self, coord):

        start_time = time.time()

        gain = self._gain * (1 - self.get_progress())
        squared_gain = gain * gain
        coord_matrix = np.array([
            [x, y] for x in self._width_range for y in self._height_range
        ]).reshape(self._width, self._height, 2)
        distance_matrix = np.subtract(coord_matrix, coord)
        squared_dist_matrix = np.multiply(distance_matrix, distance_matrix).sum(axis=2)
        activation_map = np.multiply(
            np.exp(
                np.divide(-squared_dist_matrix, squared_gain)
            ), self._learning_rate
        )

        exec_time = time.time() - start_time
        som_util.log_with_args(exec_time=exec_time, message='generate activation map')

        return activation_map

    def get_bonus_weight_map(self, error_map, activation_map):
        trained_count = 0
        start_time = time.time()

        bonus_weight_map = np.zeros(
            self._width * self._height * self._dimension
        ).reshape(self._width, self._height, self._dimension)

        for pos_x, e_col, a_col in zip(self._width_range, error_map, activation_map):
            for pos_y, feature_error, activate in zip(self._height_range, e_col, a_col):
                if activate >= self._learn_threshold:
                    bonus_weight_map[pos_x][pos_y] = np.multiply(feature_error, activate)
                    trained_count += 1

        exec_time = time.time() - start_time
        som_util.log_with_args(
            exec_time=exec_time, message='get bonus weight map',
            trained_count=trained_count)

        return bonus_weight_map

    def train_feature_vector(self, feature_vector):

        error_map = self.get_error_map(feature_vector)
        bmu_coord = self.get_bmu_coord(error_map)
        activation_map = self.get_activation_map(bmu_coord)
        bonus_weight_map = self.get_bonus_weight_map(
            error_map=error_map, activation_map=activation_map)

        start_time = time.time()

        self.map = np.add(self.map, bonus_weight_map)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            exec_time=exec_time, message='train feature vector complete.',
            progress=self.get_progress() * 100)

    def train_feature_map(self, feature_map):

        sample_count = 0
        max_sample_count = feature_map.map.__len__()

        start_time = time.time()

        for sample_unit in feature_map.map:
            self.train_feature_vector(sample_unit)

            sample_count += 1
            cycle_progress = float(sample_count) / max_sample_count * 100
            som_util.log_with_args(
                log_level='DEBUG', message='train feature finished',
                cycle_progress=cycle_progress)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            log_level='INFO', exec_time=exec_time, message='all feature trained',
            progress=self.get_progress() * 100, sample_count=sample_count)

    def train(self, feature_map):

        self._iteration = 0
        som_util.log_with_args(
            log_level='INFO', message='train start',
            iteration=self._max_iteration_count)

        start_time = time.time()

        while self.get_progress() < 1:
            self.train_feature_map(feature_map)
            self.do_progress()

        exec_time = time.time() - start_time
        som_util.log_with_args(
            log_level='INFO', exec_time=exec_time,
            message='train finished')
