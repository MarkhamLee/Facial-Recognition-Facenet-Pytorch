### Facial Recognition with Facenet-PyTorch: MTCNN & Google's InceptionResnet V1
I spent most of the winter and spring '22 researching, prototyping, and testing various approaches to building a solution that would use facial recognition for identity verification. Think: comparing photos taken of someone “the day of” with a reference photo. This repo is meant to serve as "lab notes" and/or a starting point for building similar solutions in the future.   

### Recent Updates

#### 05/15/2024 - Evaluation & Preparation Updates
* In the 'benchmarking_utils' folder there is a script for testing inference speed. It will break down the results for the face detection and embedding portions of the inference pipeline and it leaves out the first 10 images, so the slower "warm-up" inferences don’t skew the results. 

* Created a 'deployment_utilities' folder that will contain things you could use to prepare to deploy a solution like this. It currently holds a script that can generate embeddings of reference photos, so that when you are doing active photo matches you only need to run ML on day of photos and not the reference photos as well. 

#### 04/19/2024 - Security Updates
*  Updated the Dockerfile to be smaller and more secure by moving to a two-stage build. The final image is ~2.1GB smaller and has a smaller risk surface area because the tools used to build the Python dependencies are only used in the first/build stage and are then dropped from the final image, as the fully built Python dependencies are just copied over. 
* Updated base image and Python dependencies to address security issues 

#### 03/29/2024
Made several updates over the last couple of weeks 
* Real time monitoring via MQTT & InfluxDB - e.g., tracking match %, inferencing latency and similarity over time 
* Added Github Actions configs to automatically build multi-architecture images whenever new code is pusehd to the repo 
* General refactoring, cleaned up the Euclidean similarity calculation 

#### 11/10/2023
* Refactored unit tests, cleaned up code, better comments, annotations, etc.

#### 9/04/23:
* Added logging 
* Added basic unit tests
* Refactored API calls in Jupyter Notebook, refactored "test_helper" script 

### How does it work? 
#### The Short Answer:
* It is a Flask and Gunicorn based API wrapper around the [Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch), with enhancements to make it easy to demo, test or iterate on for a production level solution: monitoring of match results, separate endpoints for two photos vs. a photo and a previously generated embedding, unit tests, etc. Note: the solution is built based on the presumption that the user already has a method for capturing photos and presenting them to the API.  

* There is an endpoint for presenting two photos “photo_match” and an endpoint for presenting a pre-calculated tensor and a photo “cached_endpoint”; both endpoints return a json with details on the photo match including inferencing speed, math result and similarity score 

* You can run the API locally on your machine either by building the docker container and then running the docker container, or from the command line via: 

```
gunicorn  --bind 0.0.0.0:6000  wsgi:app
```
Note: the above command will only work on Unix based systems, e.g., Linux and OSX. I have had no problems running the container on Windows based systems, but please be aware that I do 95-98% of my development work for computer vision on Linux. 

* There is also a Jupyter notebook that's preconfigured to connect to the API, which you can use to run quick tests, demos, etc.


#### The Longer Answer 

It is a Flask, Gunicorn and PyTorch based solution for facial recognition, good enough for demos and a foundation for building something that is more production ready. While it would ideally be run on hardware with a GPU, if you are running it on hardware with compute power equivalent to an Intel 11th gen i5 it will run more than fast enough for a demo. 

If I were to implement this, my architectural approach would be to split out machine learning components that generate embeddings away from the rest of the solution and deploy it on GPU enabled hardware. I.e., the part that receives photos, retrieves reference data, and calculates similarity scores can be on slower/cheaper hardware and the dedicated machine learning components can run on more expensive GPU based hardwar  

* The server will run on port 6000, but you can specify any port you like, provided it isn't already in use. E.g., on OSX Port 5000 is reserved for Airplay   

* Once the API is running you can either ping the endpoints with custom code or use Postman to test it. If you use Postman: 

    * Body --> Form Data --> select files under key, and then use the "select files" button to select your image file or pre-computed tensors that were generated using this same model. Meaning: you would have to write some code yourself to generate and then store those tensors.  
    * You need to specify the similarity threshold as “threshold” and the match type as “cosine,” I suggest 0.4 for the threshold. Note: it will always default to cosine similarity, including type is just a placeholder that may be used for future features. 

* Presenting two photos to the /photo_match endpoint via a POST operation returns a JSON with details on the photo match  

* Presenting pre-generated tensors of the reference photo and a recent sample photo to the cached_photo endpoint does the same as the above. Note: the tensors will need to be generated with the same “device type” this solution is running on, e.g., CPU tensors if running on a CPU, GPU if running on a GPU. 

* If more than one face is detected in the sample photo, the solution will attempt to match each face in the photo to the reference photo, so try to use photos that only have one person in them. 

* Getting this close to production ready would require, at the very least:     
    * The machine learning model would need to be fine-tuned for the population the solution is being used for. I cannot stress this one enough, what is here is good enough for a demo, but the model would absolutely need to be retrained to use in a production scenario. 
    * Authentication and encryption for the API connections 
    * A proxy server and load balancer like NGINX to run in conjunction with Gunicorn. I included Gunicorn with the repo as that's just software driven, and you can run that just via doing a quick install, while Nginx requires hardware configurations/is not conducive for a quick cloning of a repo and then running a demo, so I left that out.  
    * You would need to build out the infrastructure for capturing photos and storing cached photos, tensors, etc. 

### Technical Details - What's Under The Hood: 
* Facenet-PyTorch is deployed via a Flask and Gunicorn API based Docker Inferencing container OR you can run the Gunicorn server directly on your machine IF it is a Unix based OS like OSX or Ubuntu, Windows users will need to run the docker container to use this.  
* MTCNN hyper-parameters were tweaked to increase face detection accuracy  
* PyTorch's built in functions for cosine and Euclidean distance were used to calculate photo similarity.  
* An ideal implementation would use reference photos for which embeddings have already been generated, and then compare those photos to a sample photo.  

* This was originally built on a MacBook running OSX and then tested (for GPU acceleration) on Ubuntu 20.04 with an NVIDIA GPU; in recent months I've also successfully tested it/run it on the following hardware and OS combinations: 
    * **Intel x86 devices:** running Ubuntu 22.04 with and without an NVIDIA GPU 
    * **ARM Devices:** Raspberry Pi 4B: Raspbian, Ubuntu; Orange PI 5 Plus: Armbian, Ubuntu; Orange Pi 3B: Armbian; Libre Le Potato running Raspbian.  
    * **AMD64 devices:** 5560U running Ubuntu 22.04. 

* If you use this on a Windows device you would have to test via running the docker container, you cannot just test the individual components like Guinicorn + Flask without using Docker as gunicorn only runs on Unix based operating systems. 

### API Details 
   * /ping just returns an "OK" if the API is running properly  
   * Hitting the /photo_match endpoint with a pair of photos (reference or known photo vs a sample photo) returns a JSON with the following information: 
        * Match success (yes/no) 
        * Cosine distance, where <0.4 = a match 
        * Face Detection probabilities for the face detection part of the detection pipeline 
        * Inferencing latency: how long did it take to match the photos  
   * The “/cached_data” endpoint allows you to pass a photo along with a set of tensors for a cached or pre-processed reference photo 

### Included files 
* server.py runs the Flask API - see the "short answer" under "How does it work?" for the command line instructions 
* photo_inferencing.py - contains the code that runs the machine learning models 
* score_service.py calculates the distance or similarity scores 
* Running the docker file will build a container and then you can test/experiment via hitting the API with Postman or custom code. 


### Acknowledgements 
This project is heavily influenced by [Tim Esler's Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch) a PyTorch based implementation of Google's Facenet research paper, which is in turn heavily influenced by [David Sandberg's TensorFlow implementation](https://github.com/davidsandberg/facenet) of same. 

### References 
* [Machine Learning Mastery - How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) 