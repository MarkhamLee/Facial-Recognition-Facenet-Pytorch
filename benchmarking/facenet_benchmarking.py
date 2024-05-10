# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Flask based API wrapper around the Facenet-PyTorch facial recognition library
import os
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from statistics import mean, stdev
from PIL import Image
from time import time

from common.logging_util import LoggingUtilities  # noqa: E402


class FacenetBenchmarking:

    def __init__(self, photo_path: str, tensor_path: str):

        self.logger = LoggingUtilities.\
            console_out_logger("facenet_benchmarking")

        # check for cuda
        self.cuda_check()

        # get models
        self.mtcnn, self.resnet = self.get_models()\

        # get files
        self.photo_files = self.get_file_lists(photo_path)
        self.tensor_files = self.get_file_lists(tensor_path)

        self.logger.info('file lists created')

        # run benchmarking tests
        latency_list, all_latency_list = self.run_tests(self.photo_files,
                                                        self.tensor_files)

        # calculate stats
        self.calculate_stats(latency_list, all_latency_list)

    def cuda_check(self):

        self.device =\
            torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.logger.info('Running on device: {}'.format(self.device))

    def get_models(self):

        mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7],
                      0.709, True, True, None, False, device=self.device)

        # Instantiate Resnet for Facial Geometry (Embeddings)
        resnet = InceptionResnetV1(pretrained='vggface2',
                                   classify=True).eval().to(self.device)

        return mtcnn, resnet

    def get_file_lists(self, path: str):

        file_list = []

        for (dirpath, dirnames, filenames) in os.walk(path):
            filenames.sort()
            file_list += [os.path.join(dirpath, file) for file in filenames]

        return file_list

    def run_tests(self, photo_files, tensor_files):

        latency_list = []
        all_latency_list = []

        # to exclude the first 10 photos from the avg latency calculation
        # gives the model time to warm-up
        count = 0

        # to keep this simple, just timing how long it takes to attempt
        # to match a pair of photos, w/o considering accuracy and the like.
        for photo, tensor in zip(photo_files, tensor_files):

            start = time()
            # generate embedding from the photo
            sample_photo = Image.open(photo)
            sample_cropped = self.mtcnn(sample_photo).to(self.device)
            sample_embeddings = self.resnet(sample_cropped.unsqueeze(0))

            # load reference embedding
            reference_embedding = torch.load(tensor)

            # compare
            cosd = F.cosine_similarity(reference_embedding, sample_embeddings)
            score = (1 - cosd.item())  # noqa: F841

            end = time()

            latency = round((end - start), 2) * 1000
            count += 1

            all_latency_list.append(latency)

            if count > 10:
                latency_list.append(latency)

        return latency_list, all_latency_list

    def calculate_stats(self, latency_list: list, all_latency_list):

        total_photo_pairs = len(latency_list)
        avg_latency = round(mean(latency_list), 2)
        stdev_latency = round(stdev(latency_list), 2)

        all_latency_mean = round(mean(all_latency_list), 2)

        self.logger.info(f'The average latency was: {avg_latency}ms over {total_photo_pairs} photos, with a standard deviation of: {stdev_latency}')  # noqa: E501
        self.logger.info(f'Average latency including the warm-up runs was: {all_latency_mean}ms')  # noqa: E501

        # note: warm-up runs is for GPU not CPU, as the first couple of
        # inferences are slower as the GPU acceleration software is
        # figuring out the best way to run the calculations on the GPU.


test = FacenetBenchmarking("test_photos/", "cached_cpu_tensors/")
