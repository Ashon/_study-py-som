
from som_profiler import SomProfiler
from som_profiler import FeatureMapProfiler
import som_util

def main():

    # 200,000,000,000
    width = 25
    height = 25
    som_info = {
        'width': width,
        'height': height,
        'dimension': 450,
        'randomize': True,
        'gain': 20,
        'max_iteration': 100,
    }

    som = SomProfiler(**som_info)

    sample_length = 10000
    sample_map = FeatureMapProfiler(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)

    som.train(sample_map)

    for sample in sample_map.map:
        som_util.print_simmap(som, sample, width, height)

if __name__ == '__main__':
    main()
