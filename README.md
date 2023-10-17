## Facial Recognition with Facenet-PyTorch: MTCNN  & Google's InceptionResnet V1
I spent most of the winter and spring '22 researching, prototyping and testing various approaches to building a solution that could be used for identity verification, where you would take a known photo of someone and compare it to a photo taken as they attempt to enter a high security area. This repo is meant to serve as "lab notes" and a canned demo/starting point for building something like this for someone else in the future, I.e. if nothing else this will save me from having to go through various directories on my dev box eight months from now, trying to figure out what version is the "the one". I.e., it's a starting point for future projects that I may iterate on from time to time.  

<br>

## How does it work? 
**The short answer:** 
* It's a Flask API and Gunicorn based wrapper around the [Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch), with a number of functional and machine learning based enhancements to make an easy to depoloy/use it as a facial identity solution, I.e. I took it a bit further and added cosine and euclidian distance calculations to determine if two photos are of the same individual. Note: the solution is built based on the presumption that the user already has a method for capturing photos and presenting them to the API. 
* This isn't production ready (yet), as it would need some additional work around the server (load balancing, proxy), logging, decoupling the front end API endpoints from the ML components and ideally would be fine tuned for the population/persons it was being used for. 
* You can run the API locally on your  machine either by building the docker container and then running the docker container, or from the command line via: gunicorn  --bind 0.0.0.0:6000  wsgi:app
* The server will run on port 6000, but you can specify any port you like, as long as it isn't already in use. E.g., on OSX Port 5000 is reserved for Airplay  
* Once the API is running you can either ping the endpoints with custom code or use Postman to test it. If you use Postman:
    * Body --> Form Data --> select files under key, and then use the "select files" button to select your image file or pre-computed tensors that were generated using this same model. Meaning: you'd have to write some code yourself to generate and then store those tensors. 
* note: for threshold I suggest 0.4 for cosine distance. You specify threshold as "threshold" in your API call, and the scoring type as "type", for now there is just cosine distance, will add other distance measures in the future. Meaning: it doesn't currently matter what distance measure you specify it will always be cosine. 

**The long answer:**
* presenting two photos to the /photo_match endpoint via a POST operation returns a JSON with details on the photo match 
* Presenting pre-generated tensors of the reference photo and a recent sample photo to the cached_photo endpoint does the same as the above 
* If more than one face is detected in the sample photo, the solution will attempt to match each face in the photo to the reference photo 
* The default matching method is cosine distance, plan to add Euclidian distance in the near future 
* The "distance" features are provided via built in functions of PyTorch 
* Matches are calculated via taking a pair of embeddings and calculting the "distance" between them using cosine distance 

<br>

## Technical Details - What's Under The Hood: 
* Facenet-PyTorch is deployed via a Flask and Gunicorn API based Docker Inferencing container OR you can run the gunicorn server directly on your machine IF it's a Unix based OS like OSX or Ubuntu, Windows users will need to run the docker container to use this. 
* MTCNN hyper-parameters were tweaked to increase face detection accuracy 
* PyTorch's built in functions for cosine and Euclidan distance were used to calculate photo similarity. 
* An ideal implementation would use reference photos for which embeddings have already been generated, and then compare those photos to a sample photo. 
* You can run as many workers as your hardware can support for the gunicorn/entrypoint server, but be aware that this would create separate instances of the ML models, which would take up a lot of memory. I.e. for a production level instance you'd need to decouple the front end that receives photos from the ML piece, but it depends on how many matches you're doing per a given unit of time. 
* This was built on a MacBook running OSX, but I also tested it on Ubuntu 20.04 
* If you use this on a windows device you would have to test via running the docker container, you can't just test the individual components like guinicorn + flask separately without using Docker as gunicorn only runs on Unix based operating systems. 


<br>

### API Details 
   * /ping just returns an "OK" if the API is running properly 
   * Hitting the /photo_match endpoint with a pair of photos (reference or known photo vs a sample photo) returns a JSON with the following information:
        * Match success (yes/no)
        * Cosine distance, where <0.4 = a match
        * Face Detection probabilities for the face detection part of the detection pipeline
        * Inferencing latency: how long did it take to match the photos 
    
   * the cached_data endpoint allows you to pass a photo along with a set of tensors for a cached or pre-processed reference photo 

<br>

### Included files 
* server.py runs the Flask API - see the "short answer" under "How does it work?" for the command line instructions 
* photo_inferencing.py - contains the code that runs the machine learning models 
* score_service.py calculates the distance or similarity scores 
* Running the docker file will build a container and then you can test/experiment via hitting the API with Postman or custom code. 

### Updates 9/04/23:
* Added logging 
* Added basic unit tests
* Refactored API calls in Jupyter Notebook, refactored "test_helper" script 

### Updates 6/11/23: 
* Added Jupyter Notebook as a "front-end" allows you to easily input photos and see results, including 
basic UX as far as green or red tint over photos depending on whether or not they match 
* Code clean-up, refactoring of methods, in particular simplifying the methods supporting each API endpoint 

### Updates 06/07/2023: 
* Code clean-up, fixed a few issues 
* Noticed some weirdness with the Euclidian calculations, removed it for now, will re-add in the future 
* Cleaned up the Readme to hopefully make using this more clear 

### Updates 6/04/2023: a few small changes 
* Added ability to specify distance threshold 
* Added ability to specify which type of scoring you want to use, I.e., cosine distance or Euclidian distance/L2 
* Some small refactoring here and there 

### Future updates 
* ~~Add logging: record each incoming API request, messages @ various points for tracking potential errors~~ 
* ~~Add a Jupyter Notebook "front-end" for easy testing/use without Postman~~ added 6/11/23
* ~~Add test scripts~~ 
* Add additional similarity scores/distance measures
* Add authentication  
    

## Acknowledgements 
This project is heavily influenced by [Tim Esler's Facenet-PyTorch repo](https://github.com/timesler/facenet-pytorch) a PyTorch based implementation of Google's Facenet research paper, which is in turn heavily influenced by [David Sandberg's TensorFlow implementation](https://github.com/davidsandberg/facenet) of same. 

## References 
* [Machine Learning Mastery - How to Develop a Face Recognition System Using FaceNet in Keras](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) 