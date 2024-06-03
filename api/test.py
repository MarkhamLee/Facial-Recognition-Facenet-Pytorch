# Unit tests via the unit test library
import requests
import json
import unittest
import tracemalloc
tracemalloc.start()


class TestLogConfiguration(unittest.TestCase):

    # class method so that variables and (future) classes/libraries only
    # have to be declared once
    @classmethod
    def setUpClass(self):

        # URL to check that the API is working properly
        self.health = 'http://0.0.0.0:6000/ping'

        # a pair of photos to be verified
        self.photo_pair = 'http://0.0.0.0:6000/identity'

        # a photo and a previously computed tensor of the reference photo
        self.photo_tensor = 'http://0.0.0.0:6000/cached_data'

        # define our test photos
        self.reference = 'images/Allyson_Felix_0001.jpg'
        self.evaluated = 'images/Allyson_Felix_0002.jpg'

        self.reference_a = 'images/George_Karl_0001.jpg'
        self.evaluated_a = 'images/Gregg_Popovich_0002.jpg'

        self.reference_b = 'cpu_tensors/Aaron_Sorkin_0001.pt'
        self.evaluated_b = 'images/Aaron_Sorkin_0002.jpg'

    # testing the API health endpoint
    def test_health(self):

        response = requests.request("GET", self.health)
        parsedResponse = json.loads(response.text)
        status = parsedResponse['API Status']

        self.assertEqual(status, 200, 'API not running')

    # testing the endpoint you present two photos to for verification
    def test_twoPhotos(self):

        payload = {'type': "cosine", 'threshold': 0.35}

        files = {'reference': open(self.reference, 'rb'),
                 'sample': open(self.evaluated, 'rb')}

        headers = {}

        response = requests.post(self.photo_pair, headers=headers,
                                 data=payload, files=files)

        # make sure the files are closed
        for file in files.values():
            file.close()

        # convert the JSON string to a python dictionary
        response = json.loads(response.text)

        self.assertEqual(response['match_status'], 1,
                         "The Match Status is Wrong")
        self.assertEqual(response['score'], 0.215,
                         "The Cosine Distance is wrong")
        self.assertEqual(response['score_type'], 'cosine',
                         "The score type is wrong")
        self.assertEqual(response['score_threshold'], 0.35,
                         "The score threshold is wrong")
        self.assertIsNotNone(response['inferencing_latency(ms)'],
                             "The latency data is missing")

    # testing the two photo endpoint with two photos that don't match
    def test_non_match(self):

        payload = {'type': "cosine", 'threshold': 0.35}

        files = {'reference': open(self.reference_a, 'rb'),
                 'sample': open(self.evaluated_a, 'rb')}

        headers = {}

        response = requests.post(self.photo_pair, headers=headers,
                                 data=payload, files=files)

        for file in files.values():
            file.close()

        # convert the JSON string to a python dictionary
        response = json.loads(response.text)
        self.assertEqual(response['match_status'], 0,
                         "The Match Status is Wrong")
        self.assertEqual(response['score'], 0.518,
                         "The Cosine Distance is wrong")
        self.assertEqual(response['score_type'], 'cosine',
                         "The score type is wrong")
        self.assertEqual(response['score_threshold'], 0.35,
                         "The score threshold is wrong")
        self.assertIsNotNone(response['inferencing_latency(ms)'],
                             "The latency data is missing")

    # testing the endpoint that you present a photo and a cached/stored set of
    # tensors representing the reference photo
    def test_cached_data(self):

        payload = {'type': "cosine", 'threshold': 0.35}

        files = {'reference': open(self.reference_b, 'rb'),
                 'sample': open(self.evaluated_b, 'rb')}

        headers = {}

        response = requests.post(self.photo_tensor, headers=headers,
                                 data=payload, files=files)

        for file in files.values():
            file.close()

        # convert the JSON string to a python dictionary
        response = json.loads(response.text)

        self.assertEqual(response['match_status'], 1,
                         "The Match Status is Wrong")
        self.assertEqual(response['score'], 0.28,
                         "The Cosine Distance is wrong")
        self.assertEqual(response['score_type'], 'cosine',
                         "The score type is wrong")
        self.assertEqual(response['score_threshold'], 0.35,
                         "The score threshold is wrong")
        self.assertIsNotNone(response['inferencing_latency(ms)'],
                             "The latency data is missing")


if __name__ == '__main__':
    unittest.main()
