/*
Global variables, constants & imports
@Author Maxime Touroute
April 2015
*/

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define FACEDETECT_SEARCH_RECT_OFFSET 20
#define CAMSHIFT_BACKPROJ_THRESHOLD 100
#define TRACKWINDOW_OFFSET 20
#define NUMBER_OF_LETTERS_IN_NEURAL_NETWORK 26

int DETECTION_MODE;
#define FACEDETECT_MODE 1
#define CAMSHIFT_MODE 2
#define HANDDETECT_MODE 3

int PROGRAM_MODE;
#define SIGN_TRAINING_MODE 1
#define SIGN_RECOGNITION_MODE 2

// Displayed letter when training / recognizing hand signs
char letterDisplayed = '.';
int frameCounter = 0;
string cascadeName = "./data/cascades/haarcascade_frontalcatface.xml";

#ifdef __APPLE__
const char *SCRIPT_SHUFFLE_PATH = "./scripts/osx/shuffle.sh";
const char *SCRIPT_BRACKET_REMOVER_PATH = "./scripts/osx/bracketRemover.sh";
#else
const char *SCRIPT_SHUFFLE_PATH = "./scripts/linux/shuffle.sh";
const char *SCRIPT_BRACKET_REMOVER_PATH = "./scripts/linux/bracketRemover.sh";
#endif