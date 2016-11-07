# Sign Language Detect

This project was developped as part of a Computer Vision course to demonstrate the workflow of a basic machine learning approach. In a single app, you can train your computer to recognize your own hand signs language, and launch an automatic hand sign language recogniser based on your training data.

# How to use the app
* Find a well lit room with a uniform background
* launch the app and put your left hand at the same height of your head. The app will try to detect it
  * you can press [TAB] to reset the detection if it fails
* make a hand sign, and press the corresponding alphabet letter to generate a training set
  * the default app expects you to train four letters
* to make your neural network as robust as possible, repeat the operation multiple times for each letter
* switch to recognition mode by pressing [SPACE]
* a neural network based on your very own training dataset recognizes your hand sign


# Example : training the letters B, C and K

The UI is a simple overlay of your webcam video stream. Once you make a hand sign and press the corresponding letter, the trained letter is displayed in the upper-right corner :

![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/train_B.png)

When doing so, the app creates a mask of the image based on your skin tones :

![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/backproj_full_B.png)

Repeat the process to generate a training dataset for the neural network. Here's what a trained dataset for the letters B, C, and K looks like :

![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/backprojs_B.jpg)
![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/backprojs_C.jpg)
![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/backprojs_K.jpg)

Once the dataset is created, press [SPACE] to train the neural network and switch to recognition mode. You can test the efficiency of your dataset immediately :

![](https://github.com/maximetouroute/Sign-Language-Detector/blob/master/img/recog_B.png)

# Installation

* OSX specific install:

Install homebrew and execute those commands :
```
brew install coreutils
brew install gnu-sed
```

* Linux and OSX:

Execute those commands to launch the app :
```
cmake .
make
./Main
```

# Algorithms Flow

## Algorithm flow for Hand Detection

* Face detection is made through a Haar Cascade detector
* once the face is detected, skin color is analyzed to set-up a tracking algorithm based on skin color (very fast)
* the tracking algorithm tracks the skin face once, then moves its track area to find the left hand
* if the hand is lost, the algorithm restarts

## Algorithm Flow for Training Mode

 * User makes a hand sign and hits the corresponding alphabet key
 * the hand is extracted from the frame
 * a mask of the hand is created, resized as a square, and exported in a training data file
 * external scripts are called to format and shuffle training data

## Algorithm Flow for Recognition Mode

 * The neural network is trained based on your dataset, and loaded back in the app as a predictor
 * prediction is executed on the video stream every 10 frames
 * the recognized letter is displayed on the upper-right of the viewport


# Code structure

code is intentionnaly structured by course chapters, in a chain architecture

Main ---> Language Detection (Chapter 3) --> Camshift (Chapter 2) ---> FaceDetect (Chapter 1)
