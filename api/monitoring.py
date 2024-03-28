# (C) Markham 2022 - 2024
# Facial-Recognition-Facenet-Pytorch
# Flask based API wrapper around the Facenet-PyTorch facial recognition library
# Utilities for instrumentation/reporting the like
import os
import uuid
from paho.mqtt import client as mqtt
from logging_util import logger


class ReportingCommunication:

    def __init__(self):

        # load variables
        self.load_variables()

        # get MQTT clientID
        self.clientID = str(uuid.uuid4())

        # get MQTT client
        self.client = self.mqttClient()

    # Load variables
    def load_variables(self):

        self.userName = os.environ['MQTT_USER']
        self.pwd = os.environ['MQTT_SECRET']
        self.host = os.environ['MQTT_BROKER']
        self.port = os.environ['MQTT_PORT']

    # Generate MQTT client
    def mqttClient(self):

        def connectionStatus(client, userdata, flags, code):

            if code == 0:
                logger.info('connected to MQTT broker')

            else:
                logger.debug(f'connection error occured, return code: {code}, retrying...')  # noqa: E501

        client = mqtt.Client(self.clientID)
        client.username_pw_set(username=self.userName, password=self.pwd)
        client.on_connect = connectionStatus

        code = client.connect(self.host, self.port)

        # this is so that the client will attempt to reconnect automatically/
        # no need to add reconnect logic.
        client.loop_start()

        return client, code
