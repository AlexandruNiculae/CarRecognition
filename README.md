# CarRecognition      


## Requirements:
  - [Pillow][pillow_link]
  - [Tensorflow][tensorflow_link]
  - [Keras][keras_link]


## Objective:
Compare the differences between Tensorflow and Keras when it comes to Convolutional Networks. Both libraries are used for Machine Learning and other Artificial Intelligence related research.
The question we are trying to answer is which one is more programmer friendly and which one is more efficient or more powerful.

In order to compare them on a different and much more complicated dataset rather than MNIST, we will use a dataset of cars in which the network is tasked with recognizing the car's producer from the image.

***

## Convolutional Networks:
  Here we are comparing the Convolutional Networks provided by each tutorial    

## From the tutorial:  

  Here we want to see what are the results from the guides that each of the frameworks provide.  

1.Tensorflow:  
  According to [this guide][tf_guide], Tensorflow obtains an average of 99.2% on the MNIST dataset.  

2.Keras:  
  The [Keras guide][k_guide] results in an accuracy of over 99% for MNIST.   

## State of the art  
  The dataset we used was part of [FGComp 2013][FGCComp2013], in 2 tracks:  
      - In Track 1, bounding boxes will be provided at both training and test time.  
      - In Track 2, bounding boxes will be provided only at training time.  
   The team **Inria-Xerox** used visual features based on dense SIFT and RGB descriptors, spatial coordinates coding, and Fisher Vectors. Then, "one versus all" SVM classifiers are run to predict the category of each image.  
   On track 1 they reach the accuracy of 87.7876 and on track 2 82.7136 accuracy.  
     
   The **Symbiotic** team used a method in which Fisher-encoded SIFT and color histogram are extracted from the foreground area and each of the detected parts. All features are concatenated together into the final high dimensional representation, which is fed into a linear SVM for classification. 5-fold bagging is used for track 1 in the linear SVM stage.  
   On track 1 they reach the accuracy of 81.0347 and on track 2 77.9878 accuracy.  
   
  
## Our Results:  
Here are the results that we obtained using the [Stanford dataset of cars][cars_data] instead of MNIST.


1.Tensorflow:  
  **TBC**

2.Keras:  
  **TBC**


---
# Team Info:
## To do:
- [x] Save dataset in project files
- [x] Export .mat to something more `pythonish`
- [x] Parse pictures to both nets in the same way
- [ ] Execute both nets for the same amount of time
- [ ] Conclude results

---


## Conclusions:  
After this experiment these are our conclusions regarding each of the technologies in which we worked.  

  [tensorflow_link]:https://www.tensorflow.org/
  [keras_link]: https://keras.io/
  [pillow_link]: http://pillow.readthedocs.io/en/4.3.x/
  [tf_guide]: https://www.tensorflow.org/get_started/mnist/pros
  [k_guide]: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
  [cars_data]: http://ai.stanford.edu/~jkrause/cars/car_dataset.html
  [FGCComp2013]:https://sites.google.com/site/fgcomp2013/
