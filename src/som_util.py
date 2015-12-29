
import logging
import numpy as np
import sys

logging.basicConfig(
    filename='som.log',
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

def log_with_args(log_level='DEBUG', instance=None, exec_time=None, message='', **kwargs):

    loggers = {
        'CRITICAL': logging.critical,
        'DEBUG': logging.debug,
        'ERROR': logging.error,
        'WARN': logging.warn,
        'INFO': logging.info,
    }

    logger = loggers.get(log_level)

    log_body = []
    if instance:
        obj_log = '{type}@{hash}'.format(
            type=type(instance), hash=instance.__hash__()
        )
        log_body.append(obj_log)

    if exec_time:
        exec_log = '[{exec_time:.3f} sec]'.format(exec_time=exec_time)
        log_body.append(exec_log)

    if message:
        log_body.append(message)

    args_log = ' '.join([
        '[{key}={value}]'.format(key=key, value=kwargs[key]) for key in kwargs
    ])

    log_body.append(args_log)
    logger(' '.join(log_body))

def print_simmap(som, sample, width, height):
    sample_error_map = np.subtract(som.map, sample)
    sample_error_map = np.sum(
        np.multiply(sample_error_map, sample_error_map), axis=2)

    sample_max_err = np.max(sample_error_map)

    sample_a_sim_map = np.divide(sample_error_map, sample_max_err)

    print '--' * width

    for pos_x in range(width):
        for pos_y in range(height):
            i = sample_a_sim_map[pos_x][pos_y]
            if 1 < i:
                mark = 'X'
            elif i == 1:
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
