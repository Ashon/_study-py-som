
import time
import numpy as np

from som import Som
from som import FeatureMap
import som_util


class FeatureMapProfiler(FeatureMap):

    def __init__(self, width=0, height=0, dimension=0, randomize=False):
        super(FeatureMapProfiler, self).__init__(
            width=width, height=height,
            dimension=dimension, randomize=randomize)

        som_util.log_with_args(
            log_level='INFO', instance=self, dimension=self._dimension,
            width=self._width, height=self._height)

    def get_error_map(self, feature_vector):

        start_time = time.time()

        error_map = super(FeatureMapProfiler, self).get_error_map(feature_vector)

        exec_time = time.time() - start_time
        som_util.log_with_args(exec_time=exec_time, message='get error map')

        return error_map

    @staticmethod
    def get_bmu_coord(error_map):

        start_time = time.time()

        bmu_coord = FeatureMap.get_bmu_coord(error_map)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            exec_time=exec_time, message='found bmu',
            x=bmu_coord[0], y=bmu_coord[1])

        return bmu_coord


class SomProfiler(Som):

    def __init__(self, width=0, height=0, dimension=0, randomize=False,
                 threshold=0.5, learning_rate=0.05, max_iteration=5, gain=2):

        super(SomProfiler, self).__init__(
            width=width, height=height, dimension=dimension, randomize=randomize,
            threshold=threshold, learning_rate=learning_rate,
            max_iteration=max_iteration, gain=gain)

        som_util.log_with_args(
            log_level='INFO', instance=self, threshold=self._threshold, gain=self._gain,
            learning_rate=self._learning_rate, iteration=self._max_iteration_count
        )

    def get_activation_map(self, coord):

        start_time = time.time()

        activation_map = super(SomProfiler, self).get_activation_map(coord)

        exec_time = time.time() - start_time
        som_util.log_with_args(exec_time=exec_time, message='generate activation map')

        return activation_map

    def get_bonus_weight_map(self, error_map, activation_map):

        start_time = time.time()

        bonus_weight_map = super(SomProfiler, self).get_bonus_weight_map(error_map, activation_map)

        exec_time = time.time() - start_time

        trained_count = np.where(bonus_weight_map.ravel() != 0)[0].__len__()
        som_util.log_with_args(
            exec_time=exec_time, message='get bonus weight map',
            trained_count=trained_count)

        return bonus_weight_map

    def train_feature_vector(self, feature_vector):

        start_time = time.time()

        super(SomProfiler, self).train_feature_vector(feature_vector)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            exec_time=exec_time, message='train feature vector complete.',
            progress=self.get_progress() * 100)

    def train_feature_map(self, feature_map):
        sample_count = feature_map.map.__len__()
        start_time = time.time()

        super(SomProfiler, self).train_feature_map(feature_map)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            log_level='INFO', exec_time=exec_time, message='all feature trained',
            progress=self.get_progress() * 100, sample_count=sample_count)

    def train(self, feature_map):

        som_util.log_with_args(
            log_level='INFO', message='train start',
            iteration=self._max_iteration_count)

        start_time = time.time()

        super(SomProfiler, self).train(feature_map)

        exec_time = time.time() - start_time
        som_util.log_with_args(
            log_level='INFO', exec_time=exec_time,
            message='train finished')
