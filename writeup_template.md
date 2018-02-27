# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/lenet_5.png "LeNet Model"
[image2]: ./examples/nvidia_model.png "Nvidia Model"
[image3]: ./examples/center_example.jpg "Center example"
[image4]: ./examples/left_example.jpg "Left Example"
[image5]: ./examples/right_example.jpg "Right Example"
[image6]: ./examples/near_side.jpg "Near side"
[image7]: ./examples/centered.jpg "Centered"
[image8]: ./examples/flipped.png "Flipped"


#### Project Files

The project includes the following files:
* model.py containing the script to create and train the model. The file shows the pipeline implemented for training and validating the model, and it contains comments to explain how the code works.
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Model architecture

Initially, I started with the original LeNet architecture (shown in the following image) and I decided to use the following data set https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

![alt text][image1]

LeNet performs well in the first curve of the track but right before the bridge, it fails and the car drops to the water. It was recommended to use the model created by Nvidia and explained in http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

![alt text][image2]

The model consists on the following layers:
* Data normalization
* 5x5 convolution 2x2 stride activation=Relu
* 5x5 convolution 2x2 stride activation=Relu
* 5x5 convolution 2x2 stride activation=Relu
* 3x3 convolution
* 3x3 convolution
* Fully connect layer Output=100 Drop=0.5
* Fully connect layer Output=50 Drop=0.5
* Fully connect layer Output=10 Drop=0.5
* Fully connect layer Output=1 Drop=0.5

Normalization layer: a keras lambda layer is used to normalized the data. Since pixel range is (0, 255), the images is normalized by dividing by 255 and subtracting 0.5, so the final normalized pixel will have a value in a range of (0, 0.5).

The model includes RELU layers to introduce nonlinearity in the 5x5 Convolutions.

The fully connect layers have 0.5 drops to reduce overfitting.

The NVidia model resulted in a much better result. The car is able to drive autonomously in the track. The model is implemented in the nvidia_model_define() of the model.py script.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting in the fully connect layers. 

The model was trained and validated on different data sets to ensure that the model was not overfitting, the data is shuffled as well. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I chose batch size of 128, it works pretty well.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road (https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip). After some exploration of the sample data, it used a combination of center lane driving combined with recovering from the left and right sides of the road, which the vehicle to know what to do when it is near of the sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the LeNet-5 model. I thought this model might be appropriate because it has been demonstrated to work well on another problems, so I decided to give it a try. However, it does not work well. As recommendation in the course, I tried a powerful network architecture (the NVidia model) which works better for this particular problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I included dropout layers to the original model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

Here is an example image of center lane driving:

![alt text][image3]

Initially, I used just the center driving images, but the result was not good. The vehicle was not selecting the the right angle to turn on the curves. So, I also used the images of the left and right driving, that gave more data and also it helped to the car to predict a better angle on the curve.

![alt text][image4] ![alt text][image5]

It is important to record the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer the angle correctly when it is near the sides. An example of recovering is shown in the following images:

![alt text][image6]
![alt text][image7]

To augment the data sat, I also flipped images and angles thinking that this would help to generalize the model and reduce overfitting. An example is shown

![alt text][image8]

After the collection process, I had around 40k of data points. I then preprocessed this data by normalizing the data an cropping the images. Cropping the images helps to focus the model on the important details of the road (sides, curves) and it won't distract in things like trees. It helped to reduce loss faster and in less number of epochs.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
