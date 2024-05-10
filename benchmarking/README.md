## Benchmarking & Testing

Benchmarking script to assess performance on various types of hardware, the general idea is that this would be for a scenario where you have "day of" photos that you are comparing to pre-computed embeddings to speed up inferencing. There are sample photos and pre-computed tensors in the tensor folders, all you have to do is run the facenet_benchmarking script, wait a few seconds and you will see the stats in console: mean + standard deviation. 

A couple of key details:

* The cosine distance calculation + loading the tensor only takes a few milliseconds(often less than one), so you can assume that if you were to use two photos for the comparison it would take twice as long as it would if you were to use cached/pre-computed tensors. 
* Be mindful of not using GPU tensors when running on a CPU and vice-versa, they are calculated slightly differently and will not be compatible with the hardware you are using. 
*I ignore the first 10 runs to give the testing a "warm-up" period, as the first handful of inferences are typically slower on a GPU while the GPU acceleration software/drivers are figuring out the most efficient way to perform the calculations on the GPU hardware. This wouldn't be relevant if you ran the test on a CPU, but I left it in there anyway for consistency's sake. 
