
import time

from som import Som
from som import FeatureMap
from som_util import print_map


def main():
    width = 50
    height = 50
    som_info = {
        'width': width,
        'height': height,
        'dimension': 10,
        'randomize': False,
        'gain': 30,
        'max_iteration': 100,
        'learning_rate': 0.3
    }

    print('Initialize SOM => %s' % ''.join([
        '[%s=%s]' % (key, som_info[key]) for key in som_info.keys()
    ]))
    som = Som(**som_info)

    sample_length = 20
    print('Initialize Samples [sample_length=%s]' % sample_length)
    sample_map = FeatureMap(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)

    total_time = 0

    print('Train Start')
    while som.get_progress() < 1:

        start_time = time.time()
        som.train_feature_map(sample_map)
        train_execution_time = time.time() - start_time

        som.do_progress()

        total_time += train_execution_time

        print((
            'Train Result'
            f' [progress={som.get_progress():.3f} %]'
            f' [exec_time={train_execution_time:.3f} sec]'
        ))

    print('Train Complete.')
    print(f'Total Exec [{total_time:.3f} sec]')
    print(f'Iteration Count [{som.get_iteration_count()}]')
    print(f'Exec AVG [{total_time / som.get_iteration_count():.3f} sec]')

    for idx in range(len(sample_map.map)):
        print_map(som, sample_map, idx)


if __name__ == '__main__':
    main()
