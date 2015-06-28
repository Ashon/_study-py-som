
import logging

logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

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
