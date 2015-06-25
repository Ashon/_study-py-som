
'''
    self organizing map

    @author ashon
'''

import logging
import time

import numpy as np

logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

def clamp(value, min_value, max_value):
    return min_value if value < min_value else max_value if max_value < value else value

class FeatureMap(object):

    def __init__(self, width=0, height=0, dimension=0, randomize=False):

        if randomize:
            fill_method = np.random.rand
        else:
            fill_method = np.zeros

        self.map = fill_method(width * height * dimension).reshape(width, height, dimension)

        self._width = width
        self._height = height
        self._dimension = dimension

        logging.info('Featuremap@{hash} [width={width}][height={height}][dimension={dimension}]'.format(
            hash=self.__hash__(), dimension=self._dimension,
            width=self._width, height=self._height))

    def get_bmu_coord(self, feature_vector):
        ''' returns best matching unit's coor
            @complexity
                O((width * height) * (3 * dimension + 2))
        '''

        start_time = time.time()

        # O(width * height * dimension)
        error_list = np.subtract(self.map, feature_vector)

        # O(width * height * dimension)
        squared_error_list = np.multiply(error_list, error_list)

        # O(width * height * dimension)
        sum_squared_error_list = np.sum(squared_error_list, axis=2)

        # O(width * height)
        min_error = np.amin(sum_squared_error_list)

        # O(width * height)
        min_error_address = np.where(sum_squared_error_list == min_error)

        bmu_coord = np.array([min_error_address[0][0], min_error_address[1][0]])

        exec_time = time.time() - start_time
        logging.debug('[{exec_time:.3f} sec] found bmu[x={x}][y={y}]'.format(
            exec_time=exec_time, x=bmu_coord[0], y=bmu_coord[1]))

        return bmu_coord

    def get_bmu(self, feature_vector):
        ''' returns bmu '''
        bmu_coord = self.get_bmu_coord(feature_vector)

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

        logging.info('Som@{hash} [threshold={threshold}][learning_rate={learning_rate}][gain={gain}][iteration={iteration}]'.format(
            hash=self.__hash__(), threshold=self._threshold, learning_rate=self._learning_rate,
            gain=self._gain, iteration=self._max_iteration_count))

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

    def _get_activation_map(self, coord):
        '''
            @complexity
                O(4 * width * height)
        '''

        start_time = time.time()

        gain = self._gain * (1 - self.get_progress())
        squared_gain = gain * gain

        # O(width * height)
        coord_matrix = np.array([
            [x, y] for x in range(self._width) for y in range(self._height)
        ]).reshape(self._width, self._height, 2)

        # O(width * height)
        distance_matrix = np.subtract(coord_matrix, coord)

        # O(width * height)
        squared_dist_matrix = np.multiply(distance_matrix, distance_matrix).sum(axis=2)

        # O(width * height)
        activation_map = np.multiply(
            np.exp(np.divide(-squared_dist_matrix, squared_gain)), self._learning_rate
        )

        exec_time = time.time() - start_time
        logging.debug('[{exec_time:.3f} sec] generate activation map [progress={progress}%]'.format(
            progress=self.get_progress(), exec_time=exec_time))

        return activation_map

    def train_feature_vector(self, feature_vector):
        '''
            @complexity
                O((width * height) * (4 * dimension + 7))
        '''

        start_time = time.time()

        # O((width * height) * (3 * dimension + 2))
        bmu_coord = self.get_bmu_coord(feature_vector)

        activation_map = self._get_activation_map(bmu_coord)

        # O(width * height)
        negative_map = -self.map
        error_map = np.add(negative_map, feature_vector)

        trained_count = 0
        # O(width * height * dimension)
        for x, error_col, activate_col in zip(range(self._width), error_map, activation_map):
            for y, feature_error, activate in zip(range(self._height), error_col, activate_col):
                if activate >= self._learn_threshold:
                    bonus_weight = np.multiply(feature_error, activate)
                    self.map[x][y] = np.clip(
                        np.add(self.map[x][y], bonus_weight), a_min=0, a_max=1)

                    trained_count += 1

        exec_time = time.time() - start_time
        logging.debug('[{exec_time:.3f} sec] train feature vector [progress={progress}%] [trained_count={count}]'.format(
            progress=self.get_progress(), exec_time=exec_time, count=trained_count))

    def train_feature_map(self, feature_map):
        start_time = time.time()

        sample_count = 0
        max_sample_count = feature_map.map.__len__()

        for sample_unit in feature_map.map:
            self.train_feature_vector(sample_unit)

            sample_count += 1
            cycle_progress = float(sample_count) / max_sample_count * 100
            logging.debug('train feature finished [cycle_progress={cycle_progress}%]'.format(
                progress=self.get_progress(), cycle_progress=cycle_progress))

        exec_time = time.time() - start_time
        logging.debug('[{exec_time:.3f} sec] all feature trained [progress={progress}%] [sample_count={count}]'.format(
            progress=self.get_progress(), exec_time=exec_time, count=sample_count))

    def train(self, feature_map):
        start_time = time.time()
        logging.info('train start [iteration={iteration}]'.format(
            iteration=self._max_iteration_count))

        self._iteration = 0

        while self.get_progress() < 1:
            self.train_feature_map(feature_map)
            self.do_progress()

        exec_time = time.time() - start_time
        logging.info('[{exec_time:.3f} sec] train finished'.format(
            exec_time=exec_time))
