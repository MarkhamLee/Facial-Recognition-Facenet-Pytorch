# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Example of how to generate reference tensors
import os
import sys
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_util import LoggingUtilities  # noqa: E402
from common_utils.general_utilities import GeneralUtils  # noqa: E402


class CacheTensors():

    def __init__(self, photo_path: str, cache_path: str,
                 device=None):

        self.logger = LoggingUtilities.\
            log_file_logger("tensor_caching")

        self.utils = GeneralUtils()

        if not device:
            # set device based on best available - e.g., CUDA if
            self.device = self.set_device()

        else:
            self.device = device

        self.logger.info(f'Running on device: {device}')

        self.mtcnn, self.resnet = self.get_models()

        file_list, name_list = GeneralUtils.get_file_list(photo_path)

        self.generate_tensors(file_list, name_list, cache_path)

    def set_device(self):

        if torch.cuda.is_available():
            self.device = 'cuda:0'

        else:
            self.device = 'cpu'

        return self.device

    def get_models(self):

        mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7],
                      0.709, True, True, None,
                      False, device=self.device).eval()

        self.logger.info("MTCNN Loaded")

        # Instantiate Resnet for Facial Geometry (Embeddings)
        resnet = InceptionResnetV1(pretrained='vggface2',
                                   classify=True).eval().to(self.device)
        self.logger.info("InceptionResnetV1 Loaded")

        return mtcnn, resnet

    def generate_tensors(self, photo_files, name_list, save_path):

        for photo, file_name in zip(photo_files, name_list):

            photo = Image.open(photo)

            # face detection
            cropped_photo = self.mtcnn(photo).to(self.device)
            # generate tensor
            embedding = self.resnet(cropped_photo.unsqueeze(0))

            tensor_name = (f'{file_name}.pt')

            # save tensor
            self.utils.save_pytorch_tensors(embedding, save_path, tensor_name)


save_tensors = CacheTensors("../benchmarking/test_photos/",
                            "facenet_tensor_cache", "cpu")
