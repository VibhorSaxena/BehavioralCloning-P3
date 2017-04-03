# Behavioral Cloning (Assignment-3)
---

## 1) Description of the pipeline
---

The pipeline can be described in the following phases:- 

1) Data gathering

2) Data Augmentation

3) Model Design

4) Model Training and Validation

5) Running the simulation

---

The individual steps of the pipeline are explained below:-

---

### 1) Data gathering

A lot of emphasis has been given on data collection to induce the desired behaviour in the car :-

1) 2 laps of center lane driving

![Alt text](./images/center_lane.jpg?raw=true "Center Lane Driving")

2) 2 laps of reverse lane driving

![Alt text](./images/reverse_lane.jpg?raw=true "Reverse Lane Driving")

3) 1 lap of recovery driving (Going to the edge of the road and then starting to record the car coming to the center)

![Alt text](./images/recovery_driving.jpg?raw=true "Recovery Driving")

4) 2 laps of driving on the second track

![Alt text](./images/track2.jpg?raw=true "Track 2")


---

### 2) Data Augmentation

I have augmented the data in many ways so that the model is able to generalize better and doesn't overfit to the data.

Following steps have been used:-

1) Flipping images:- All images have been flipped with a negative steering angle so that the car doesn't try to keep turning in one direction.

![Alt text](./images/flipped.png?raw=true "Flipped Image")

2) Using left and right camera images:- I have used the center, left and right camera images in a 60:20:20 ratio randomly taken. This ensures better tackling of the recovery problem. I experimented with various values of the steering angle correction for left and right images and found that the car was driving the most sanely when I used the value of 0.15.

(Images from center, left and right camera)


![Alt text](./images/center_left_right.png?raw=true "Images from center, left and right camera")

3) Changing the brightness values:- I have converted the image into the HSV space and then changed the value of V by a factor of randomly chosen values from (0.5,1.2). This ensures that the car would be able to drive in different lighting conditions as well. 

![Alt text](./images/brightness_augmented.png?raw=true "Brightness Augmented Image")


I had implemented a generator as per the requirement but I found that it was not necessary since my aws g8 instance was being able to store the data in memory itself and using a generator was turning out to be quite slow because of regular fetching from disk.
Therefore the code for the generator has been commented out.


---
### 3) Model Design

I progressively started with a very simple model utilizing only fully connected linear units, moving on to LeNet and further more complex models. To introduce more non-linearities in the model, I have tried to test two models:-

1) Training Nvidia's End to End Model for self driving cars suggested [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

2) Finetuning Vgg 16 model by just keeping last two layers trainable and adding the Dense layers for regression instead of classification.

The two model's architectures are mentioned below:-

Nvidia           | VGG
:-------------------------:|:-------------------------:
![](./images/nvidia.png?raw=true)  |  ![](./images/vgg.png?raw=true)


In practice, I found that the NVidia model took much lower training and prediction time than the Vgg model while achieving similar results, so I have finally gone ahead and made the submission for the Nvidia's model.

#### Description of the model:-

The Nvidia's model starts of with a lambda layer which performs normalization followed by a cropping layer
This is followed by 3 layers of 5x5 convolutional layers followed by 2 3x3 layers with relu activation. 
This are followed by 4 fully connected layers with relu activation and a final output linear neuron. 
Every layer has a dropout of 0.25 to avoid overfitting.


---

---
### 4) Model Training and Validation

The model defined mean square error as the target metric and used adam's optimizer to find calculate the weights.

I tried various values of batch size and epochs and realized that model was overfitting with too many epochs so I finally just went ahead with a batch size of 32 and 2 epochs.

---

### 5) Running the simulation

The results of the simulation are shown below in gifs (hyperlinked to their respective videos). The results were very satisfactory on track 1. The car was behaving well on track 2 as well initially but it ends up bumping into a very steep curve so improvements can be made on the second track.


Track1           | Track2
:-------------------------:|:-------------------------:
[![alt text](https://j.gifs.com/oYW7JA.gif)](https://youtu.be/IdNtVQe9-zg) | [![alt text](https://j.gifs.com/qjWDLD.gif)](https://youtu.be/bsskv5Vf8hg) 