# Sign Language Detect Project

This project was developped as part of a Computer Vision course to demonstrate the workflow of a basic machine learning approach. In a single app, you can train your computer to recognize your own hand signs language, generate a neural network based on your training data, and launch an automatic hand sign language recognition based on your training data.

# How to use the app
* Find a place well lit with a uniform background
* Launch the app and put your left hand at the same height of your head. The app will try to detect it.
  * You can press [TAB] to reset the detection if it failed
* make a hand sign, and press the corresponding alphabet letter to generate a training set
  * The default app expects you to train four letters.
* To make your neural network as robust as possible, repeat the operation multiple times for each letter
* Switch to recognition mode by pressing [SPACE]
* A neural network based on your very own training dataset recognizes your hand sign



# Example : training the letters B, C and K

The UI is a simple overlay of your webcam video stream. Once I make a hand sign and press the corresponding letter, the trained letter is displayed in the upper-right corner

![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/train_B.png)

When doing so, the app creates a mask of the image based your skin tones

![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/backproj_full_B.png)

By pressing repeating the process, I generate training dataset for the neural network. Here's what a train dataset for the letters B, C, and K looks like :

![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/backprojs_B.png)
![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/backprojs_C.png)
![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/backprojs_K.png)

Once this dataset is created, pressing [SPACE] trains the neural network, and use it for recognition. You can test the efficiency of your dataset immediately :  

![](https://github.com/maximetouroute/Video-Stabilisation-For-Soccer-Game/blob/master/img/backprojs_K.png)

# How to use it

* OSX
Install homebrew
```
brew install coreutils
brew install gnu-sed
```

* Linux and osx :
Execute those commands to launch the app
‘‘‘
cmake .
make
./Main
‘‘‘



# Algorithms Flow

## Algorithm flow for Hand Detection

* Face detection is made through a Haar Cascade detector
* once the face is detected, skin color is analyzed to set-up a tracking algorithm based on skin color (very fast)
* the tracking algorithm tracks the skin face once, then moves its track area to find the left hand
* if the hand is lost, the algorithm restarts

 ---- Hit [SPACE] to switch between SIGN_TRAINING_MODE and SIGN_RECOGNITION_MODE
 ---- Hit [TAB] to reset the algorithm if camshift tracking is lost

## Algorithm Flow for Training Mode

 * The user makes a hand sign and its the corresponding alphabet key
 * the hand is extracted from the frame, and enlarged a bit
 * a mask of the hand is created, resized as a square, and exported as a text file
 * external scripts are called to format and shuffle the training data

## Algorithm Flow for Recognition Mode

 * the neural network is trained based on your data, and loaded back in the app
 * Every 10 frames, hand sign prediction is executed on video stream
 * the recognized letter is displayed on the upper-right of the viewport


# Code structure

code is intentionnaly structured by course chapters, in a chain architecture

Main ---> Language Detection (Chapter 3) --> Camshift (Chapter 2) ---> FaceDetect (Chapter 1)
