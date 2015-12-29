
from som_profiler import SomProfiler
from som_profiler import FeatureMapProfiler
import som_util

def main():

    width = 100
    height = 100
    dimension = 10000

    som = SomProfiler(**{
        'width': width,
        'height': height,
        'dimension': dimension,
        'randomize': True,
        'gain': 20,
        'max_iteration': 100,
    })

    sample_length = 15
    sample_map = FeatureMapProfiler(
        width=sample_length, height=1,
        dimension=dimension, randomize=True)

    som.train(sample_map)

    for sample in sample_map.map:
        som_util.print_simmap(som, sample, width, height)

if __name__ == '__main__':
    main()
