# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/histraw.png "Raw histogram"
[image2]: ./images/histaug.png "Augmented histogram"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./images/visualize.gif "Normal Image"
[image7]: ./images/model.png "Model diagram"

## Rubric Points

### Project files

The project includes the following files:
* ```modelzoo.py``` contains all the models experimented with for this project
* ```drive.py``` for sending steering commands to the simulator in the autonomous mode
* ```model.h5``` containing a trained convolution neural network
* ```augment.py``` functions for our data augmentation pipeline
* ```visualize.py``` to visualize network activations 
* This file, ```writeup.md```, summarizing the results

#### Usage

Using the Udacity simulator and the ```drive.py``` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

### Data collection and augmentation 

We start with examining the dataset that was already provided. The following image shows the histogram of the steering commands in the dataset.

![Histogram of raw data][image1]

Clearly, the dataset is unbalanced. There are disproportionately high number of images corresponding to zero or very small steering commands. Since this track goes counter-clockwise, most of the steering commands are to the left.  

Another problem with data is the almost non-existence of data for very high steering commands. These high steering samples are encountered not only along sharp curves but also during recovery from deviations. This was kept in mind during data collection. For the final model we had close to 15,000 data points (45,000 images including the left and the center cameras). This dataset consisted of:
1. Dataset already provided with the project
2. Intermittent data from 2 laps of driving in the other direction
3. Intermittent data from driving around curves, critical points
4. Intermittent data recovering from large deviations from the center, both along straight parts and curves. 

#### Generator and the augmentation pipeline

To mitigate the problems with the data that we discuss above, we implemented a generator function for use with ```fit_generator``` in Keras. The generator works as follows:
1. Choose a random index from the available data
2. Check if the steering command for this index is close to zero
2. For that index, choose a random camera between left, center and right
3. Load the image and the corresponding steering command
4. Transform this image randomly through the data augmentation pipeline
5. Go back to 1. till we have ```batch_size = 128``` number of images
6. Return this set of images

Following are the features of the pipeline
1. **Drop images with small steering commands**
The ```augment.augDrop(thres, prob)``` helps drop images with steering command smaller than ```thres``` with a probability of ```prob```. We used ```prob=0.975```.  The implementation ensures that a set of images are not completely ignored. Different images are taken in each time and are augmented in different ways by the following pipeline. 
2. **Use left, center and right camera images**
For each image in the batch, a random camera is chosen and the corresponding steering angle is correction for being "off-center". This is not physically accurate but works when the data is this scarce. 
3. **Random flips**
Each image is randomly flipped horizontally with a settable probability and its steering command is negated. The function ```augment.augFlip`` is the implementation. 
4. **Random brightness transform**
The function ```augment.augBright``` is used to brighten or dim the image by a factor chosen randomly from a uniform distribution. 
5. **Random x and y translations**
The function ```augment.augTranslate``` does this for us. Translations in the y-axis are also accompanied by a corresponding correction in the steering command.  

The result of this pipeline  for 10 epochs on training is shown below. 
![alt text][image2]

This is the accumulated histogram after 7 epochs of training. The data is now much better distributed than before. 

### Model Architecture and Training Strategy

#### 1. Model selection

A lot of model architectures were attempted for this project. These networks have been implemented in the ```modelzoo.py``` file. The table shows a summary.
 

Though all of them worked for this task, our particular case is much simpler than real world driving that nVidia and comma.ai models are designed for. For this reason and considering training resources available, we sought a smaller and simpler model with a focus on track 1 performance.

We started with a working modification of the LeNet-5 model, implemented as ```mLeNet``` in ```modelzoo.py``` simplifying it. The following though processes was used,
1. Experimentation showed grayscale models worked as well as colour images for this task. We therefore used a single channel input to our model. The S channel in the HSV colorspace performed the best.
2. Full uncropped images required more dense layers. Cropping the top and bottom parts of the images to select only the road not only helped reduce the input size but also allowed removing some dense layers.
3. One the dense part of the network was smaller it was much easier to visualize the activation of the convolutional layer. With that as a guide, we reduced the number of filters in that layer.   

The most important criteria, of course, was the driving performance on track 1. Although we continuously looked at the test and validation loss values, they are no indication of the driving performance. As long as they decreased with the epochs and are close together, we were not bothered with the exact numbers.

Following are the details of our final model tuned model, implemented as ```mSmall``` in ```modelzoo.py```. It has around 1.3k trainable parameters.

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Remarks
====================================================================================================
Normalize (Lambda)               (None, 32, 64, 1)     0           Normalization layer
____________________________________________________________________________________________________
Conv (Convolution2D)             (None, 28, 60, 3)     78          3 filters, 5x5, valid, ELU
____________________________________________________________________________________________________
MaxPool (MaxPooling2D)           (None, 14, 30, 3)     0           size = stride = (2,2), valid
____________________________________________________________________________________________________
Flatten (Flatten)                (None, 1260)          0           
____________________________________________________________________________________________________
Dropout (Dropout)                (None, 1260)          0           keep_prob = 0.5
____________________________________________________________________________________________________
Output (Dense)                   (None, 1)             1261        Final continous output
====================================================================================================  
Total params: 1,339
```

Or pictorially, 

![alt text][image7]


In short, the model consists of a convolution neural network with 5x5 filter sizes and simple single dense layer with one output. The convolution layer uses ELU as it's activation function although RELU yielded a similar performance too. 

#### 2. Reducing overfitting

These models almost always overfit. This problem is compounded by the fact that we have a very small number of parameters. We tried to generalize it as follows:
1. A dropout layer was introduced after flattening the convolution layer.
2. Data augmentation was used to try and ensure it performs well to a larger range of inputs. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The ```keep_prob``` in the dropout layer was continuously reduced to prevent overfit while still having it to perform on track 1. The final value was ```0.5```.

### Performance and analysis

The video of the model driving on track 1 is available [here](./models/run.mp4) and the saved model is available [here]('./models/model.h5)

While our model worked very well for track 1, it did not work well for certain points on track 2. Our analysis is the following, although we did not the time and resources at hand to implement the suggestions.
1. Track 2 has sharper curves of the magnitude not encountered in the training data. So a few laps of training with track 2 data will help. 
2. Very large slopes on track 2 reduce the road visibility. The crop window needs to be tuned to accommodate this.
3. Our model is probably too small to generalize to this track. There might be a need to go for a bigger model.
 

However, this is one advantage of a small as small as ours. It is easier to not only visualize the activations, but  also trace the activations back to the input image. We did this as follows (implemented with ```visualize.py```)
1) Obtained the model weights from the trained model
2) Built the second model that is exactly like our original model but terminated it right after the Flatten layer.
3) Element-wise multiply the output with the weights to create a "contributions matrix" that represents the amount each pixel contributes to the final steering command.
4) Resize this contributions matrix to the input image size and overlay it on the input. The resulting image gives a visualization of both positive and negative contributions by pixels in the image. 

Following is a small subclip of the full video. The full visualization is [here]('models/visualize.mp4').

![alt text][image6]

Ignoring the constantly changing background, it is very clear that the model has learnt weigh the pixels along the lane boundary the most .

The background problem in the implementation is due to normalization. Contribution map is normalized per frame and this causes the background to vary sharply across frame. We can avoid this by using a global normalization.  