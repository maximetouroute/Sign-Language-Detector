/*
TP3
Language Detection
@Author Maxime Touroute
April 2015
*/

#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include "camshift.cpp"
int SCREENSHOT_VALUE;
// g++ -ggdb `pkg-config --cflags opencv` -o `basename handdetect.cpp .cpp` handdetect.cpp `pkg-config --libs opencv`

typedef struct NeuralData
{
    CvANN_MLP mlp;
    int classCount;

} NeuralData;

NeuralData theNeuralData;

void exportLetterInTrainingFile(cv::Mat& m, const char letter);
void drawText(cv::Mat& frame, string text, int fontSize, int xpos, int ypos, CvScalar color);
void recognize(cv::Mat frame, struct camshiftData& data);
void train(cv::Mat frame, struct camshiftData& data, int keyboardKey);
void initNeuralData();


/*
@param mask: the mask to dilate
@param dilationSize: the size of the dilation
*/
void dilateMask(Mat& mask, int dilationSize)
{
    int dilation_type = MORPH_RECT;
    Mat dilate_element = getStructuringElement( dilation_type, Size( 2 * dilationSize + 1, 2 * dilationSize + 1 ),
                         Point( dilationSize, dilationSize ) );
    dilate( mask, mask, dilate_element );
}

void initNeuralData()
{
    theNeuralData.mlp.load("./data/neural/neuralData.xml");
    theNeuralData.classCount = 26;
    SCREENSHOT_VALUE = 0;
}

void train(cv::Mat frame, struct camshiftData& data, int keyboardKey)
{ 
    // expand a bit trackWindow
    int newX = data.trackWindow.x - TRACKWINDOW_OFFSET;
    int newY = data.trackWindow.y - TRACKWINDOW_OFFSET;
    int newWidth = data.trackWindow.width + 2 * TRACKWINDOW_OFFSET;
    int newHeight = data.trackWindow.height + 2 * TRACKWINDOW_OFFSET;
    if( newX < 0 )
    {
        newX = 0;
    } 
    if( newY < 0 )
    {
        newY = 0;
    } 
    if( newX + newWidth > frame.cols )
    {
        newWidth = frame.cols - newX;
    } 
    if( newY + newHeight > frame.rows )
    {
        newHeight = frame.rows - newY;
    }
    Rect newTrackWindow = Rect(newX, newY, newWidth, newHeight);

    // get hand backproj, resize to a 16x16 patch and export to train file
    Mat exportBackproj = data.backproj(newTrackWindow); 

    // Dilate backproj to get better results
    dilateMask(exportBackproj, 3);
    imwrite( "./data/backproj" + std::to_string(SCREENSHOT_VALUE) + ".jpg", exportBackproj ); // TODO: export multiple backproj to show behaviour
    SCREENSHOT_VALUE++;
    resize(exportBackproj, exportBackproj, Size(16,16), 0, 0, INTER_LINEAR);
    exportLetterInTrainingFile(exportBackproj, keyboardKey);
    cv::imshow( "Saved Letter", exportBackproj ); 
}

void recognize(cv::Mat frame, struct camshiftData& data)
{
    CvANN_MLP mlp;
    const int classCount = 26;

    mlp.load("./data/neural/neuralData.xml");

    // expand trackWindow
    int newX = data.trackWindow.x - TRACKWINDOW_OFFSET;
    int newY = data.trackWindow.y - TRACKWINDOW_OFFSET;
    int newWidth = data.trackWindow.width + 2*TRACKWINDOW_OFFSET;
    int newHeight = data.trackWindow.height + 2*TRACKWINDOW_OFFSET;
    if( newX < 0 ) 
    {
        newX = 0;
    }
    if( newY < 0 ) 
    {
        newY = 0;
    }
    if( newX + newWidth > frame.cols ) 
    {
        newWidth = frame.cols - newX;
    }
    if( newY + newHeight > frame.rows ) 
    {
        newHeight = frame.rows - newY;
    }

    Rect newTrackWindow = Rect(newX, newY, newWidth, newHeight);
    Mat ret = data.backproj(newTrackWindow);
        dilateMask(ret, 3);
    // Resize backproj to fit trained data
    resize(ret, ret, Size(16,16), 0, 0, INTER_LINEAR);
    Mat newFrame = ret.reshape(0,1);

    // convert frame
    newFrame.convertTo(newFrame, CV_32FC1);
    CvMat frame2 = newFrame;
    CvMat* mlp_response = cvCreateMat( frame2.rows, classCount, CV_32F );
    CvPoint max_loc = {0,0};
    // Predict hand sign
    mlp.predict(&frame2, mlp_response);
    cvMinMaxLoc(mlp_response, 0, 0, 0, &max_loc, 0);
    int bestClass = max_loc.x + 'A';
    letterDisplayed = (char) bestClass;
}

void exportLetterInTrainingFile(cv::Mat& m, const char letter)
{
    Mat frame = m.reshape(0,1);
    ofstream os("./data/neural/letters_exported.txt", ios::out | ios::app);
    os<< (char) toupper(letter) << ",";
    os << frame << std::endl;
    os.close();
    // format and shuffle data
    system(SCRIPT_BRACKET_REMOVER_PATH);
    system(SCRIPT_SHUFFLE_PATH);
    printf("exported");
}

