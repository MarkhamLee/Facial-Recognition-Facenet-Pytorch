# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Flask based API wrapper around the Facenet-PyTorch facial recognition library
import torch
import warnings
from facenet_pytorch import MTCNN, InceptionResnetV1
from logging_util import logger

warnings.filterwarnings('ignore')


# face detection & embeddings
class Inferencing:

    def __init__(self):

        # this enables this class to run on GPU when available without having
        # to update the code
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')
        logger.info('Running on device: {}'.format(self.device))

        # Instantiate face detection class
        self.mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7], 0.709, True, True,
                           None, False, device=self.device)

        # Instantiate Resnet for Facial Geometry (Embeddings)
        self.resnet = InceptionResnetV1(pretrained='vggface2', classify=True).\
            eval().to(self.device)

    def identity_verify(self, reference: object, sample: object) -> object:
        self.reference = reference
        self.sample = sample

        # detect faces, generate cropped photos
        # need to update to generate a cropped photo for each face
        reference_cropped = self.mtcnn(self.reference).to(self.device)
        sample_cropped = self.mtcnn(self.sample).to(self.device)

        embeddings_reference = self.resnet(reference_cropped.unsqueeze(0))
        embeddings_sample = self.resnet(sample_cropped.unsqueeze(0))

        logger.debug('Embeddings generated for photo pair')

        return embeddings_reference, embeddings_sample

    def cached_reference(self, sample: object) -> object:

        self.sample = sample

        # detect the face in the image
        sample_cropped = self.mtcnn(self.sample).to(self.device)

        # generate embeddings
        embeddings_sample = self.resnet(sample_cropped.unsqueeze(0)).\
            detach().cpu()

        logger.debug('Embeddings generated for single photo/cached tensor workflow')  # noqa: E501

        return embeddings_sample
