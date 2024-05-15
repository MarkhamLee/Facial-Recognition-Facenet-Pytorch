import os
from torch import save as ts
from common_utils.logging_util import LoggingUtilities

logger = LoggingUtilities.console_out_logger("general utils")


class GeneralUtils():

    def __init__(self):

        pass

    @staticmethod
    def get_file_list(path: str):

        file_list = list()

        for (dirpath, dirnames, filenames) in os.walk(path):
            filenames.sort()
            file_list += [os.path.join(dirpath, file) for file in filenames]

        return file_list

    # save PyTorch tensors
    @staticmethod
    def save_pytorch_tensors(tensor: object, path: str):

        ts(tensor,  path)

        logger.info(f'Tensor saved to: {path}')
