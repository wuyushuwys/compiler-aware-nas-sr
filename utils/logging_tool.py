import logging
import time
import os


class StringColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LoggingTool:
    def __init__(self, file_path, verbose):
        self.verbose = verbose
        self.colors = StringColors()
        self.time = time.localtime
        os.makedirs(file_path, exist_ok=True)
        logging.basicConfig(
            filename=f"{file_path}/result.log",
            filemode='w',
            level=[logging.WARNING, logging.INFO, logging.DEBUG][verbose],
            format='%(asctime)s:%(levelname)s:%(message)s',
        )

    def info(self, string, is_print=True, device=0):
        if device == 0:
            if is_print:
                print(f"{self.time_updater()} INFO:{string}")
            logging.info(string)

    def warning(self, string, device=0):
        if device == 0:
            print(f"{self.colors.WARNING}{self.time_updater()} WARNING: {string}{self.colors.ENDC}")
            logging.warning(string)

    def time_updater(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', self.time())
