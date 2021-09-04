import logging.config
import yaml
import os
import queue
import multiprocessing as mp
from multiprocessing.queues import Queue
from datetime import datetime

log = logging.getLogger(__name__)

class IterableQueue(Queue):
    """An iterable queue data structure"""
    def __init__(self):
        super().__init__(ctx=mp.get_context())

    def __iter__(self):
        while True:
            try:
               yield self.get_nowait()
            except queue.Empty:
               return

def timeit(func, name):
    """Logs the execution time of a function"""
    start = datetime.now()
    result = func()
    end = datetime.now()
    timedelta = (end - start).total_seconds()
    log.info(f"Execution of {name} took {timedelta}s")
    return result

def setup_logging(
    default_path='logging.yml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)