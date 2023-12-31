import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings
warnings.filterwarnings('ignore')


# face detection & embeddings
class inferencing:

    def __init__(self):

        # this enables this class to run on GPU when available without having
        # to update the code
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')
        print('Running on device: {}'.format(self.device))

        # Instantiate face detection class
        self.mtcnn = MTCNN(160, 30, 20, [0.6, 0.7, 0.7], 0.709, True, True,
                           None, False, device=self.device)

        # Instantiate Resnet for Facial Geometry (Embeddings)
        self.resnet = InceptionResnetV1(pretrained='vggface2',
                                        classify=True).eval().to(self.device)

    def identity_verify(self, reference: object, sample: object) -> object:
        self.reference = reference
        self.sample = sample

        # detect faces, generate cropped photos
        # need to update to generate a cropped photo for each face
        reference_cropped = self.mtcnn(self.reference).to(self.device)
        sample_cropped = self.mtcnn(self.sample).to(self.device)

        # generate embeddings
        embeddings_reference = self.resnet(reference_cropped.unsqueeze(0))
        embeddings_sample = self.resnet(sample_cropped.unsqueeze(0))

        return embeddings_reference, embeddings_sample

    def cached_reference(self, sample: object) -> object:

        self.sample = sample

        # detect the face in the image
        sample_cropped = self.mtcnn(self.sample).to(self.device)

        # generate embeddings
        embeddings_sample = self.resnet(sample_cropped.unsqueeze(0))

        return embeddings_sample
