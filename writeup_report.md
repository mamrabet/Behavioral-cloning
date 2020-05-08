# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the architecture proposed by Nvidia with the follwoing layrs
1- Convolution 5*5 24 
2- Convolution 5*5 36 
3- Convolution 5*5 48 
4- Convolution 3*3 64 
5 - Convolution 3*3 64
6- Flatten
7- Fully Connected Layer 1164
8- Fully Connected Layer 100
9- Fully Connected Layer 50
10- Fully Connected Layer 10
11- Fully Connected Layer 1

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

I also included a training data driving in the reverse direction of the simulator, and several sequences driving on the bridge and on the entry/exit of the bridge.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA Deep learning architecture (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). I thought this model might be appropriate because it is widely used in the autonomous driving industry.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model worked well on both training and validation data.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially in the bridge (entry and exit). to improve the driving behavior in these cases, I added some training sequences in the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:
1- Convolution 5*5 24 
2- Convolution 5*5 36 
3- Convolution 5*5 48 
4- Convolution 3*3 64 
5 - Convolution 3*3 64
6- Flatten
7- Fully Connected Layer 1164
8- Fully Connected Layer 100
9- Fully Connected Layer 50
10- Fully Connected Layer 10
11- Fully Connected Layer 1

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][./cnn-architecture.png]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][./center.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adapt its steering to the center of the road.


Then I repeated this process on track two in order to get more data points.

Then I added data driving in the reverse direction of the simulator to adapt to right curves.

To augment the data sat, I also flipped images and angles.

After the collection process, I had 25434 number of data points. I then preprocessed this data by normalizing and cropping the images.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The network worked well on the validation data set. The ideal number of epochs was 5 as evidenced by the video. I used an adam optimizer so that manually training the learning rate wasn't necessary.
