
from som import Som
from som import FeatureMap
import som_util

def main():

    # 200,000,000,000
    width = 25
    height = 25
    som_info = {
        'width': width,
        'height': height,
        'dimension': 3,
        'randomize': True,
        'gain': 200,
        'max_iteration': 100,
    }

    som = Som(**som_info)

    sample_length = 10000
    sample_map = FeatureMap(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)

    som.train(sample_map)

    for sample in sample_map.map:
        som_util.print_simmap(som, sample, width, height)

if __name__ == '__main__':
    main()
