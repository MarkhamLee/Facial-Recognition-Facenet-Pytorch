### Quick Notes

* You can run the API with the following command:

~~~
gunicorn  --bind 0.0.0.0:6000  wsgi:app
~~~


* The file "server_plus_monitoring.py" has a feature that sends out data related to photo matches, inferencing speed, etc., via MQTT for data collection and monitoring. I.e., working with that file will require you to have an MQTT broker set up, and setup the appropriate environmental variables for logging into the broker, create topics for receiving data, etc. To use server_plus_monitoring.py instead of the 'standard' server file, just update wsgi.py to point to it instead of server.py

