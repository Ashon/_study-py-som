
import time

from som import Som
from som import FeatureMap
import som_util
import sys
import numpy as np

def main():

    width = 10
    height = 10
    som_info = {
        'width': width,
        'height': height,
        'dimension': 10,
        'randomize': True,
        'gain': 50,
        'max_iteration': 200,
    }
    print 'Initialize SOM => %s' % ''.join([
        '[%s=%s]' % (key, som_info[key]) for key in som_info.keys()
    ])
    som = Som(**som_info)

    sample_length = 20
    print 'Initialize Samples [sample_length=%s]' % sample_length
    sample_map = FeatureMap(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)
    # print sample_map.map < 1
    total_time = 0

    print 'Train Start'
    while som.get_progress() < 1:

        start_time = time.time()
        som.train_feature_map(sample_map)
        train_execution_time = time.time() - start_time

        som.do_progress()

        total_time += train_execution_time

        # bmu_idx_list = [som.get_bmu_coord(unit) for unit in sample_map.map]

        # print 'Train Result [progress={progress:.3f} %] [exec_time={exec_time:.3f} sec]'.format(
        #     progress=som.get_progress() * 100, exec_time=train_execution_time)

        # print 'Sample idx |', ' | '.join(['{sid:3d}'.format(sid=i) for i in range(sample_length)])
        # print 'BMU idx    |', ' | '.join(['{bid}'.format(bid=(str(bid[0]) + '-' + str(bid[1]))) for bid in bmu_idx_list])

    print 'Train Complete.'
    print 'Total Exec [{exec_total:.3f} sec]'.format(exec_total=total_time)
    print 'Iteration Count [{iteration}]'.format(iteration=som.get_iteration_count())
    print 'Exec AVG [{exec_avg:.3f} sec]'.format(exec_avg=total_time / som.get_iteration_count())

    # for unit in som.units:
    #     print unit, '\t'.join(['{sample:.7f}'.format(sample=sample) for sample in unit.weights])

    sample_a = sample_map.map[0]
    # print sample_a

    sample_error_map = np.subtract(som.map, sample_a)
    sample_error_map = np.multiply(sample_error_map, sample_error_map)
    sample_error_map = np.sum(sample_error_map, axis=2)

    sample_max_err = np.max(sample_error_map)

    sample_a_sim_map = np.divide(sample_error_map, sample_max_err)
    # print som.map

    print '+-%s-+' % ('--' * width)
    for x in range(width):
        sys.stdout.write('| ')
        for y in range(height):
            i = sample_error_map[x][y]
            if i > 0.8:
                mark = ' '
            elif i > 0.6:
                mark = '.'
            elif i > 0.4:
                mark = '-'
            elif i > 0.2:
                mark = '*'
            else:
                mark = '#'
            sys.stdout.write(' ' + mark)
        sys.stdout.write(' |\n')
    sys.stdout.write('\n')
    print '+-%s-+' % ('--' * width)
if __name__ == '__main__':
    main()
