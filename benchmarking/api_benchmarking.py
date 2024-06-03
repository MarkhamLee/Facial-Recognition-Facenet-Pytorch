# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Benchmarking Script for API - I.e., sending tensors + photos
# to an endpoint and timing ML inference + the performance of the
# API layer. When using make sure to match the tensor type to the
# device you're using, e.g., CPU vs GPU
import json 
import os
import requests
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

    def __init__(self, photo_path: str, tensor_path: str, endpoint):

        self.logger = LoggingUtilities.\
            log_file_logger("API benchmarking")

        self.endpoint = endpoint

        # get files
        self.photo_files = self.get_file_lists(photo_path)
        self.tensor_files = self.get_file_lists(tensor_path)

        self.logger.info('File lists created')

        # run benchmarking tests
        all_api_latency_list, all_latency_list, api_latency_list, latency_list = \
            self.run_tests(self.photo_files,
                           self.tensor_files,
                           self.endpoint)

        # calculate stats
        self.calculate_stats(all_api_latency_list, all_latency_list,
                             api_latency_list, latency_list)

    def get_file_lists(self, path: str):

        file_list = []

        for (dirpath, dirnames, filenames) in os.walk(path):
            filenames.sort()
            file_list += [os.path.join(dirpath, file) for file in filenames]

        return file_list

    def run_tests(self, photo_files, tensor_files, endpoint):

        api_latency_list = []
        all_api_latency_list = []
        all_latency_list = []
        latency_list = []
   

        # used to exclude the first 10 runs to give the model time to warm-up
        # i.e., the first are slower while the GPU acceleration software
        # determines the best way to run the model
        count = 0

        # to keep this simple, just timing how long it takes to attempt
        # to match a pair of photos, w/o considering accuracy and the like.

        for photo, tensor in zip(photo_files, tensor_files):


            payload = {"type": "cosine", "threshold": 0.35}

            files = {'reference': open(tensor, 'rb'),
                    'sample': open(photo, 'rb')}          

            headers = {}

            start = time()
            # send API request
            response = requests.post(endpoint, headers=headers,
                                 data=payload, files=files)
            end = time()

            api_latency = 1000 * round((end - start), 2)

            # close files
            for file in files.values():
                file.close()

            # convert the JSON string to a python dictionary
            response = json.loads(response.text)

            inference_latency = int(response['inferencing_latency(ms)'])

            count += 1
            
            all_latency_list.append(inference_latency)
            all_api_latency_list.append(api_latency)


            # we exclude the first ten "test runs" to give the model
            # time to warm-up on GPUs.
            if count > 10:
                latency_list.append(inference_latency)
                api_latency_list.append(api_latency)


        self.logger.info('Testing complete, calculating stats....')

        return all_api_latency_list, all_latency_list, api_latency_list, latency_list

    def calculate_stats(self, all_api_latency_list: list, all_latency_list: list,
                        api_latency_list: list, latency_list: list):

        total_photo_pairs = len(latency_list)
        avg_latency = round(mean(latency_list), 2)
        stdev_latency = round(stdev(latency_list), 2)

        api_latency_mean = round(mean(api_latency_list), 2)
        stdev_api_latency = round(stdev(api_latency_list), 2)
        

        all_latency_mean = round(mean(all_latency_list), 2)
        all_api_latency_mean = round(mean(all_api_latency_list), 2)
        total_latency_mean = avg_latency + api_latency_mean
        effective_fps = round((1000 / total_latency_mean), 2)

        test_data = []
        test_data.append([total_photo_pairs, all_latency_mean,
                          avg_latency, stdev_latency, 
                          api_latency_mean, stdev_api_latency,
                          total_latency_mean, effective_fps])

        df_columns = ["total_photo_pairs",
                      "inferencing_latency_w/o_warm_up(ms)",
                      "inferencing_latency(ms)",
                      "inferencing_latency_stdev(ms)",
                      "api_latency",
                      "api_latency_stdev",
                      "total_latency",
                      "matches_per_second(fps)"]

        stats_df = pd.DataFrame(test_data, columns=df_columns)

        results = (f'Testing Results: \n{stats_df}\n')
        self.logger.info(results)
        return stats_df


test = FacenetBenchmarking("test_photos/",
                           "cached_gpu_tensors/",
                           'http://0.0.0.0:6000/cached_data')