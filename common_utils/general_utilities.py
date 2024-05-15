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
        name_list = list()

        for (dirpath, dirnames, filenames) in os.walk(path):
            filenames.sort()
            file_list += [os.path.join(dirpath, file) for file in filenames]

            # split off the file extension from each file name, as we'll
            # keep the file name when we saved the embeddings as .pt files.
            name_list += [os.path.join(os.path.splitext(file)[0]) for file in filenames]  # noqa: E501

        return file_list, name_list

    # save PyTorch tensors
    @staticmethod
    def save_pytorch_tensors(tensor: object, path: str, tensor_name):

        file_path = (f'{path}/{tensor_name}')

        ts(tensor,  file_path)

        logger.info(f'Tensor saved to: {file_path}')
