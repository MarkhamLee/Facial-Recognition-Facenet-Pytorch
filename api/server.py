# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Flask based API wrapper around the Facenet-PyTorch facial recognition library
# This script is specifically for the API, receives requests, communicates back
# to the client, etc.
import flask
import json
import torch
import time
from PIL import Image
from flask import Flask, request
from photo_inferencing import inferencing
from score_service import similarityScore
from logging_util import logger

app = Flask('Identity')

# instantiate the class with the ML models
photo_match = inferencing()

logger.info('ML models instantiated')

# instantiate the class with the scoring functionality
scoring = similarityScore()

logger.info('scoring/similarity class instantiated')


# endpoint for API health check
# the "ping" endpoint is one that is required by AWS
@app.route("/ping", methods=['GET'])
def health():

    logger.info('health check request received')

    results = {"API Status": 200}
    resultjson = json.dumps(results)

    logger.info(f'health check response: {resultjson}')

    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')


# endpoint for matches from two photos
@app.route("/identity", methods=['POST'])
def embeddings():

    score_type = request.form.get('type')
    threshold = request.form.get('threshold')

    logger.info(f'Request received @ endpoint for photo pairs, score type: {score_type}, and match threshold: {threshold}')  # noqa: E501

    # python parses the data as a string and we need it to be a float to
    # be used for scoring
    threshold = float(threshold)

    # retrieve reference photo
    ref_file = request.files['reference']

    # retrieve sample photo
    sample_file = request.files['sample']

    logger.info('parsed image data from incomimng request')

    # load photos
    ref_img = load_images(ref_file)
    sample_img = load_images(sample_file)

    # generate pair of tensors
    # timing inferencing latency defined, which is just the time for
    #  the ML code to run
    start = time.time()

    ref_tensor, sample_tensor = photo_match.\
        identity_verify(ref_img, sample_img)

    end = time.time()

    latency = 1000 * round((end - start), 2)
    logger.info(f'Facial embeddings generated, inferencing latency: {latency} ms')  # noqa: E501

    resultjson = build_response(latency, ref_tensor, sample_tensor,
                                score_type, threshold)

    # log response and send back to client
    logger.info(f'response sent back to client {resultjson}')
    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')


# endpoint for presenting a pre-processed/cached tensor and a sample photo
@app.route("/cached_data", methods=['POST'])
def cached():

    # parse score type and threshold from POST request
    score_type = request.form.get('type')
    threshold = request.form.get('threshold')

    logger.info(f'Request received @ cached data endpoint, score type: {score_type}, match threshold: {threshold}')  # noqa: E501

    # python parses the data as a string as we need it to be a float
    # to be used for scoring
    threshold = float(threshold)

    # parse and load PyTorch tensor
    ref = request.files['reference']
    cached_tensor = torch.load(ref)

    # retrieve sample photo
    sample_file = request.files['sample']
    sample_img = load_images(sample_file)

    logger.info('data parsed from incoming request')

    # generate embeddings for sample photo
    # timing inferencing latency defined, which is just the time for
    # the ML code to run
    start = time.time()

    sample_tensor = photo_match.cached_reference(sample_img)

    end = time.time()

    latency = 1000 * round((end - start), 2)

    logger.info(f'Facial embedding generated for sample photo, inferencing latency: {latency}')  # noqa: E501

    # send data to the method that does the similarity calculations
    # and builds response payload
    resultjson = build_response(latency, cached_tensor, sample_tensor,
                                score_type, threshold)

    app.logger.info(f'response sent back to client {resultjson}')

    return flask.Response(response=resultjson, status=200,
                          mimetype='application/json')


# method that aggregates data, prepares json response and sends the data
# back to the client
def build_response(latency: float, tensor1: object, tensor2: object,
                   score_type: float, threshold: float) -> dict:

    # generate score
    score = scoring.cosine_score(tensor1, tensor2)
    logger.info('similarity score calculated')

    # get match status
    status = scoring.match_status(score, threshold)

    # round match score
    score = round(score, 3)

    logger.info(f'match status calculated: {status} from a score of: {score}')

    # prepare latency message: rounding + adding units
    latency_message = str(latency) + " ms"

    # return data
    results = {"Match Status": status,
               "Score": score,
               "Score Type": 'cosine distance',
               "Score Threshold": threshold,
               "Inferencing Latency": latency_message}

    resultjson = json.dumps(results)

    return resultjson


def load_images(image):

    with Image.open(image) as photo:
        photo.load()

    app.logger.info('photo loaded')

    return photo
