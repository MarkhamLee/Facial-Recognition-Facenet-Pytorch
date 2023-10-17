import torch.nn.functional as F


class similarityScore:

    # method for calculating cosine distance between two tensors
    def cosine_score(self, reference, sample):
        self.reference = reference
        self.sample = sample

        cosd = F.cosine_similarity(self.reference, self.sample)
        score = (1 - cosd.item())

        return score

    # method for calculating match status - putting this into a separate
    # method to accomodate multiple score types
    def match_status(self, score, threshold):

        self.threshold = threshold

        if score < self.threshold:
            return 1
        else:
            return 0
