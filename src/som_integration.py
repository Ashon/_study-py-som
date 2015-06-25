
import time

from som import Som
from som import FeatureMap
import sys
import numpy as np

def main():
    def print_simmap(som, sample, width, height):
        sample_error_map = np.subtract(som.map, sample)
        sample_error_map = np.multiply(sample_error_map, sample_error_map)
        sample_error_map = np.sum(sample_error_map, axis=2)

        sample_max_err = np.max(sample_error_map)

        sample_a_sim_map = np.divide(sample_error_map, sample_max_err)

        print '--' * width

        for x in range(width):
            for y in range(height):
                i = sample_a_sim_map[x][y]
                if i == 1:
                    mark = '#'
                elif 0.98 <= i < 1:
                    mark = 'a'
                elif 0.86 <= i < 0.98:
                    mark = 'b'
                elif 0.64 <= i < 0.86:
                    mark = 'c'
                elif 0.52 <= i < 0.64:
                    mark = '+'
                else:
                    mark = '.'
                sys.stdout.write(' ' + mark)
            sys.stdout.write('\n')
        sys.stdout.write('\n')

    # 200,000,000,000
    width = 5
    height = 5
    som_info = {
        'width': width,
        'height': height,
        'dimension': 2,
        'randomize': True,
        'gain': 100,
        'max_iteration': 100,
    }

    som = Som(**som_info)

    sample_length = 100
    sample_map = FeatureMap(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)

    som.train(sample_map)

    # for sample in sample_map.map:
    #     print_simmap(som, sample, width, height)

if __name__ == '__main__':
    main()
