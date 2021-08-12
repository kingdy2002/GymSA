import logging
import numpy as np
from collections import defaultdict

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger


class Logger(object) :
    def __init__(self, console_logger) :
        self.console_logger = console_logger
        self.stats = defaultdict(lambda: [])

    def log_stat(self, key, value, t):
        self.stats[key].append((value , t))

    def print_debug(self,information) :
        self.console_logger.debug(information)

    def print_info(self,information) :
        self.console_logger.info(information)
    
    def print_error(self,information) :
        self.console_logger.error(information)   

    def reset_stat(self) :
        self.stats = defaultdict(lambda: [])

    def print_recent_stats(self):
        log_str = "Recent Stats | Episode: {:>8}\n".format(*self.stats["episode"][-1][0])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1

            item = "{:.4f}".format(np.mean([x[0] for x in self.stats[k][-1:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)