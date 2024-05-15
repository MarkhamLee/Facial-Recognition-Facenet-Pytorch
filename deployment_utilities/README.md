## Deployment Utilities - Work in Progress - More Content Pending

Scripts, files, and documentation for use in deploying and/or maintaining a facial recognition solution.  

### Current Contents

* Cached Tensors: pre-computing embeddings/tensors of *"reference photos"* for facial recognition solutions is important for performance reasons and saving $ on compute. It's simpler/faster to generate tensors/embeddings ahead of time and then store and retrieve them as needed, rather than re-running machine learning on the reference photo each time you need to do a photo match. **generate_facenet_tensors.py** takes a folder containing reference photos and a target folder to store cached embeddings as an inputs, generates embeddings and then stores them in the target folder. It can also take an optional parameter for device type (CPU or GPU), since you must generate tensors based on the device the solution will run on. 