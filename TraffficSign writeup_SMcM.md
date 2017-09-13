#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/FrequencyofTrafficSign.png "Visualization"
[image2]: ./images/GrayscaleSample.jpg "Grayscaling"
[image3]: ./images/normalisedImage.png "Normalised"
[image4]: ./GTSRB5/00015j.jpg "Traffic Sign 1"
[image5]: ./GTSRB5/02329j.jpg "Traffic Sign 2"
[image6]: ./GTSRB5/03363j.jpg "Traffic Sign 3"
[image7]: ./GTSRB5/03978j.jpg "Traffic Sign 4"
[image8]: ./GTSRB5/05312j.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the mupy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how often the particular traffic sign classes occur within the the training data

![Frequency of Traffic Sign png][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, after experimenting with grayscale, I decided not to convert the images to grayscale because the color can actually assist with sign identification, for example, red is typically associated with stop, etc.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this increases the numerical stability.  The image before and after normalisation is shown below.

![alt text][image3]

The difference between the original data set and the augmented data set is that the data has been normalised and shuffled.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 2	        | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		|  ouput 400       					            |
| Fully connected		| ouput 120      					            |
| RELU					|												|
| Fully connected		| ouput 84      					            |
| RELU					|												|
| Fully connected		| ouput 43      					            |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the AdamOptimizer optimizer.  The batch size of 128, 30 epochs.  I used a learning rate of of 0.0025.  This appeared to be the best value as lower values tended to overfit, (using droput as part of the model should help with overfitting).  I set mu to 0, (zero mean) and set sigma = .1.  The small sigma means that the distribution is very uncertain and and the optimization becomes more confident as the training progresses.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.946 
* test set accuracy of 0.933

If a well known architecture was chosen:
* What architecture was chosen? 
The LeNet architecture was chosen.
* Why did you believe it would be relevant to the traffic sign application?  
I tried it on the MNIST data set and it worked very well and traffic signs have many similarities to letters
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The validation accuracy is above the required 93% and meets the project requirements.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, converted to jpg format:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is dark.
The second image might be difficult to classify because it is very bright.
The fifth image might be difficult to classify because it is very dark.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (50km/h)  						| 
| Speed limit (50km/h)  | Speed limit (50km/h							|
| Turn left ahead		| No passing for vehicles over 3.5 metric tons	|
| Speed limit (30km/h)  | Speed limit (100km/h)				 			|
| Ahead only		    | Ahead only     							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 60%. This compares unfavorably to the accuracy on the test set of processed previously.  The images do not appear particularly to classify and I am astonished to see how badly the model performs.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is certain that is is a "Speed limit (50km/h)" sign (probability of 0.99), and the image does contain a "Speed limit (50km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| Speed limit (50km/h)   						| 
| .00     				| Priority road                             	|
| .00					| Yield					                       	|
| .00	      			| Speed limit (80km/h)  			 		    |
| .00				    | No passing for vehicles over 3.5 metric tons  |


For the second image the model is certain that is is a "Speed limit (50km/h)" sign (probability of 0.99), and the image does contain a "Speed limit (50km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (50km/h) 			                | 
| .00     				| No passing for vehicles over 3.5 metric tons	|
| .00					| Speed limit (80km/h) 				    	    |
| .00	      			| Speed limit (60km/h)				 		    |
| .00				    | Speed limit (100km/h)              			|

For the third image the model is not very certain that is is a "Turn left ahead" sign (probability of 0.46), and the image does not contain a "Turn left ahea" sign. 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .46         			| No passing for vehicles over 3.5 metric tons  | 
| .25     				| Speed limit (60km/h)							|
| .17					| Turn left ahead					    	|
| .11	      			| Keep right				 				    |
| .01				    | Road work   						|

For the fourth image the model is certain that is is a "Speed limit (100km/h)" sign (probability of 9.99451816e-01), and the image does not contain a "Speed limit (100km/h)" sign.  In comparison with the three correct images, ther is a tiny bit of uncertainty. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (100km/h)   						| 
| .00     				| Speed limit (30km/h)							|
| .00					| Speed limit (80km/h)					    	|
| .00	      			| No passing for vehicles over 3.5 metric tons	|
| .00				    | Speed limit (70km/h))   						|

For the fifth image the model is certain that is is a "Ahead only" sign (probability of 9.99999166e-01), and the image does contain a "Ahead only" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Ahead only   						            | 
| .00     				| Yield             							|
| .00					| Speed limit (80km/h)					    	|
| .00	      			| No passing for vehicles over 3.5 metric tons  |
| .00				    | Speed limit (60km/h)   						|