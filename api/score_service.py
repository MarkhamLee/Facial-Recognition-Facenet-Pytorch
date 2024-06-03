# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Flask based API wrapper around the Facenet-PyTorch facial recognition library
# This script calculates cosine distance and does the comparison to the
# provided threshold.
import torch
import torch.nn.functional as F
from logging_util import logger


class SimilarityScore:

    # method for calculating cosine distance between two tensors
    @staticmethod
    def cosine_score(reference: float, sample: float) -> float:

        cosd = F.cosine_similarity(reference, sample)
        score = round((1 - cosd.item()), 4)

        logger.info(f'Cosine distance calculated: {score}')

        return score

    # not using this at the moment
    @staticmethod
    def euclidean_distance(reference: float, sample: float) -> float:

        dist = torch.cdist(reference, sample, p=2.0,
                           compute_mode='use_mm_for_euclid_dist_if_necessary')

        # pull the float value out of the tensor object
        dist = round((dist.item()), 2)

        logger.info(f'Euclidean distance is: {dist}')

        return dist

    # method for calculating match status - putting this into a separate
    # method to accomodate multiple score types
    @staticmethod
    def match_status(score: float, threshold: float):

        if score < threshold:
            return 1
        else:
            return 0
