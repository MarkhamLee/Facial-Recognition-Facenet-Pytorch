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

        # check device type, set appropriate cuda options
        self.cuda_check()

        # load models
        self.mtcnn, self.resnet = self.load_models()


    def cuda_check(self):

        if torch.cuda.is_available():
            self.device = 'cuda:0'

            # the effectiveness of these settings can vary between models,
            # i.e., I would experiment with them on your HW.
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        else:
            self.device = 'cpu'

        logger.info(f'Running on device: {self.device}')

    
    def get_models(self):

        mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7],
                      0.709, True, True, None,
                      False, device=self.device).eval()

        # Instantiate Resnet for Facial Geometry (Embeddings)
        resnet = InceptionResnetV1(pretrained='vggface2',
                                   classify=True).eval().to(self.device)

        return mtcnn, resnet


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
            detach()

        logger.debug('Embeddings generated for single photo/cached tensor workflow')  # noqa: E501

        return embeddings_sample
