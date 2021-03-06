import logging
import logging.config


def configure_logger(name, log_path = 'log.txt'):
    log_path = '{}.log'.format(name)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '[%(levelname)s] %(module)s %(filename)s - %(funcName)s line %(lineno)d - \n%(message)s\n - (%(asctime)s)\n', 'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_path,
                'maxBytes': 1024,
                'backupCount': 3
            }
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                #'handlers': ['console', 'file']
                'handlers': ['console']
            },
            'file': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            }
        },
        'disable_existing_loggers': False
    })
    return logging.getLogger(name)

#logger = configure_logger('default', 'log13.txt')
__get_logger = {}
def get_logger(name):
    if __get_logger.get(name):
        return __get_logger[name]
    new_logger = configure_logger(name)
    __get_logger[name] = new_logger
    return new_logger

def disable_debug():
    for _logger in __get_logger.values():
        _logger.setLevel(logging.WARNING)
def enable_debug():
    for _logger in __get_logger.values():
        _logger.setLevel(logging.DEBUG)

logger = get_logger('default')