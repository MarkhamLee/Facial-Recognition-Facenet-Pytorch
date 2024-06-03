# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Benchmarking Script for ML components, i.e., MTCNN for
# face detection and InceptionResnetV1 for generating
# embeddings. Usage: just run the script, but make edits
# to this line at the bottom:
# test = FacenetBenchmarking("test_photos/", "cached_gpu_tensors/")
# if you want to point to different photos or cached tensors.
# The script will return a dataframe with your testing data in
# it as well as write it to a log file.
# This variant is when you want to run a particularly long test,
# more helpful for GPUs, as the running the model longer often =
# significantly lower inferencing times/more akin to real life
# performance
import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from statistics import mean, stdev
from PIL import Image
from time import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from common_utils.logging_util import LoggingUtilities  # noqa: E402
from common_utils.general_utilities import GeneralUtils  # noqa: E402


class FacenetBenchmarking:

    def __init__(self, photo_path: str, tensor_path: str):

        self.logger = LoggingUtilities.\
            log_file_logger("facenet_benchmarking")

        self.utilities = GeneralUtils()

        # check for cuda, configure cudnn
        self.cuda_check()

        # get models
        self.mtcnn, self.resnet = self.get_models()\

        # get files
        self.photo_files = self.get_file_lists(photo_path)
        self.tensor_files = self.get_file_lists(tensor_path)

        self.logger.info('File lists created')

        # run benchmarking tests
        latency_list, all_latency_list, \
            face_detection_list, embedding_list = \
            self.run_tests(self.photo_files, self.tensor_files)

        # calculate stats
        self.calculate_stats(latency_list, all_latency_list,
                             face_detection_list, embedding_list)

    def cuda_check(self):

        if torch.cuda.is_available():
            self.device = 'cuda:0'

            # the effectiveness of these settings can vary between models,
            # i.e., I would experiment with them on your HW.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        else:
            self.device = 'cpu'

        self.logger.info(f'Running on device: {self.device}')

    def get_models(self):

        mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7],
                      0.709, True, True, None,
                      False, device=self.device).eval()

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
        face_detection_list = []
        embedding_list = []

        # used to exclude the first 10 runs to give the model time to warm-up
        # i.e., the first are slower while the GPU acceleration software
        # determines the best way to run the model
        count = 0

        # to keep this simple, just timing how long it takes to attempt
        # to match a pair of photos, w/o considering accuracy and the like.

        count_test = 0

        while count_test < 100:

            for photo, tensor in zip(photo_files, tensor_files):

                sample_photo = Image.open(photo)

                start = time()
                # face detection
                sample_cropped = self.mtcnn(sample_photo).to(self.device)
                end_face = time()

                sample_embeddings = self.resnet(sample_cropped.unsqueeze(0))
                end_embedding = time()

                # load reference embedding
                reference_embedding = torch.load(tensor)
                # compare
                cosd = F.cosine_similarity(reference_embedding,
                                           sample_embeddings)
                score = (1 - cosd.item())  # noqa: F841
                end = time()

                total_latency = round((end - start), 2) * 1000
                count += 1

                all_latency_list.append(total_latency)

                # we exclude the first ten "test runs" to give the model
                # time to warm-up on GPUs.
                if count > 10:
                    latency_list.append(total_latency)
                    face_detection_list.append(1000 * (end_face - start))
                    embedding_list.append(1000 * (end_embedding - end_face))

            count_test += 1

        self.logger.info('Testing complete, calculating stats....')
        return latency_list, all_latency_list, face_detection_list, \
            embedding_list

    def calculate_stats(self, latency_list: list, all_latency_list: list,
                        face_detection_list: list, embedding_list: list):

        total_photo_pairs = len(latency_list)
        avg_latency = round(mean(latency_list), 2)
        stdev_latency = round(stdev(latency_list), 2)
        effective_fps = round((1000/avg_latency), 2)

        all_latency_mean = round(mean(all_latency_list), 2)

        face_detection_mean = round(mean(face_detection_list), 2)
        embedding_mean = round(mean(embedding_list), 2)

        test_data = []
        test_data.append([total_photo_pairs, all_latency_mean,
                          avg_latency, stdev_latency,
                          face_detection_mean, embedding_mean,
                          effective_fps])

        df_columns = ["photos",
                      "latency with_warmp(ms)",
                      "overall latency(ms)",
                      "overall_stdev(ms)",
                      "face_detection_latency(ms)",
                      "embedding_latency(ms)",
                      "effective_FPS"]

        stats_df = pd.DataFrame(test_data, columns=df_columns)

        results = (f'Testing Results: \n{stats_df}\n')
        self.logger.info(results)
        return stats_df


test = FacenetBenchmarking("test_photos/", "cached_gpu_tensors/")
