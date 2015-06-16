
import unittest

from som import FeatureVector
from som import FeatureMap
from som import Som
import som_util


class TestSomUtil(unittest.TestCase):

    def test_clamp(self):
        self.assertEqual(som_util.clamp(2, 0, 3), 2)
        self.assertEqual(som_util.clamp(-30, 0, 3), 0)
        self.assertEqual(som_util.clamp(30, 0, 3), 3)
        self.assertEqual(som_util.clamp(0, 0, 3), 0)


class TestFeatureUnit(unittest.TestCase):

    def setUp(self):
        self.unit = FeatureVector(dimension=10, randomize=True)

    def test_randomized_som_unit(self):
        self.assertEqual(self.unit.get_dimension(), 10)

    def test_zerofilled_som_unit(self):
        unit = FeatureVector(dimension=3, randomize=False)
        self.assertEqual(unit.get_dimension(), 3)
        self.assertEqual(unit.weights, [0, 0, 0])
        self.assertEqual(unit.weights, [0.0, 0, 0])

    def test_get_squred_error(self):
        test_iteration = 10000

        for _ in range(test_iteration):
            target = FeatureVector(dimension=10, randomize=True)

            max_error = self.unit.get_max_error()
            sqrd_error = self.unit.get_squared_error(target)

            self.assertTrue(0 <= sqrd_error <= max_error)

    def test_get_sqrd_error_exception(self):
        target = FeatureVector(dimension=4, randomize=True)

        with self.assertRaises(Exception) as context:
            self.unit.get_squared_error(target)

        self.assertTrue('list index out of range' in context.exception)


class TestSomMap(unittest.TestCase):

    def test_som(self):

        som_instance = FeatureMap(
            width=3, height=3, dimension=3, randomize=False)

        self.assertEqual(som_instance.get_scale(), 9)


class TestSom(unittest.TestCase):

    def setUp(self):
        self.som = Som(width=5, height=5, dimension=2, randomize=True)

    def test_feature_map(self):

        for unit in self.som.units:
            self.assertEqual(unit.get_dimension(), 2)

    def test_train(self):
        sample_vector = FeatureVector(dimension=2, randomize=False)
        sample_vector.set_weights([1, 1])
        for unit in self.som.units:
            print unit, unit.weights, 'SIM[{sim:.3f}]'.format(
                sim=unit.get_euclidean_similarity(sample_vector))
        self.som.train_feature_vector(sample_vector)


        for unit in self.som.units:
            print unit, unit.weights, 'SIM[{sim:.3f}]'.format(
                sim=unit.get_euclidean_similarity(sample_vector))

        bmu = self.som.get_bmu(sample_vector)
        bmu_idx = self.som.get_bmu_index(sample_vector)
        print 'BMU[%s]' % bmu_idx, bmu, bmu.weights
        print sample_vector.weights


if __name__ == '__main__':
    unittest.main()
