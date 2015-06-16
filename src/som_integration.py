
import time

from som import Som
from som import FeatureMap


def main():

    som_info = {
        'width': 20,
        'height': 20,
        'dimension': 5,
        'randomize': True,
        'gain': 30,
        'max_iteration': 30,
    }
    print 'Initialize SOM => %s' % ''.join([
        '[%s=%s]' % (key, som_info[key]) for key in som_info.keys()
    ])
    som = Som(**som_info)

    sample_length = 1000
    print 'Initialize Samples [sample_length=%s]' % sample_length
    sample_map = FeatureMap(
        width=sample_length, height=1,
        dimension=som_info['dimension'], randomize=True)

    total_time = 0

    print 'Train Start'
    while som.get_progress() < 1:

        start_time = time.time()
        som.train_feature_map(sample_map)
        train_execution_time = time.time() - start_time

        som.do_progress()

        total_time += train_execution_time

        bmu_idx_list = [som.get_bmu_index(unit) for unit in sample_map.units]

        print 'Train [progress={progress:.3f} %],[exec_time={exec_time:.3f} sec]'.format(
            progress=som.get_progress() * 100, exec_time=train_execution_time)

        # for i in range(sample_length):
            # sample_weights = ', '.join([
            #     '{sample:.3f}'.format(sample=sample)
            #     for sample in sample_map.units[i].weights
            # ])

            # bmu_weights = ', '.join([
            #     '{sample:.3f}'.format(sample=sample)
            #     for sample in som.units[bmu_idx_list[i]].weights
            # ])

            # print '[Sample={sample_index:2d}] => [BMU={bmu:4d}][similarity={bmu_sim:.3f}]'.format(
            #     sample_index=i, bmu=bmu_idx_list[i],
            #     bmu_sim=som.units[bmu_idx_list[i]].get_euclidean_similarity(sample_map.units[i]))

    print 'Total Exec[{exec_total:.3f} sec], Iteration[{iteration}], AVG[{exec_avg:.3f} sec]'.format(
        exec_total=total_time, exec_avg=total_time / som.get_iteration_count(),
        iteration=som.get_iteration_count())

    # for unit in som.units:
    #     print unit, '\t'.join(['{sample:.7f}'.format(sample=sample) for sample in unit.weights])

if __name__ == '__main__':
    main()
