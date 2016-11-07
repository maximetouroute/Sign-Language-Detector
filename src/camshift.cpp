/*
TP2 
Camshift skin detector algorithm
@Author Maxime Touroute
April 2015
*/

#include "facedetect.cpp"

typedef struct camshiftData
{
    int vmin;
    int vmax;
    int smin;
    int hsize;
    int ch[2] ;
    float hueranges[2];
    const float* phueranges;
    Mat hsv, hue, mask, hist, backproj;
    Mat maskroi;
    Mat camshiftROI;
    Rect trackWindow;
    bool isRunning;
} camshiftData;


void initCamshiftConstants(struct camshiftData& data);
void initCamshift(Mat& frame, Rect selection, camshiftData& data);
void camshiftTrack( Mat& frame, Rect selection, struct camshiftData& data );
void executeCamshift(Mat& frame, Rect selection, struct camshiftData& data);
void initHandDetection( Mat& frame, Rect selection, camshiftData& data );

void initCamshiftConstants(camshiftData& data)
{
    data.vmin = 10;
    data.vmax = 256;
    data.smin = 30;
    data.hsize = 16;
    data.ch[0] = 0;
    data.ch[1] = 0;
    data.hueranges[0] = 0;
    data.hueranges[1] = 180;
    data.phueranges = data.hueranges;
    data.isRunning = false;
}

// Reads tint of frame seletion, and makes it the tint to track
void initCamshift(Mat& frame, Rect selection, camshiftData& data)
{
    cvtColor(frame, data.hsv, COLOR_BGR2HSV);
    // Probability map
    inRange(data.hsv, Scalar(0, data.smin, MIN(data.vmin,data.vmax)), Scalar(180, 256, MAX(data.vmin, data.vmax)), data.mask);
    //Use only the hue value
    data.hue.create(data.hsv.size(), data.hsv.depth());
    mixChannels(&data.hsv, 1, &data.hue, 1, data.ch, 1);
    
    Mat temp(data.hue, selection);
    data.camshiftROI = temp.clone();
    Mat temp2(data.mask, selection);
    data.maskroi = temp2.clone();

    // Histogram computatoin
    calcHist(&data.camshiftROI, 1, 0, data.maskroi, data.hist, 1, &data.hsize, &data.phueranges);
    normalize(data.hist, data.hist, 0, 255, CV_MINMAX);
    data.trackWindow = selection;
    data.isRunning = true;
}

void executeCamshift(Mat& frame, Rect selection, camshiftData& data)
{
    if( !data.isRunning ) 
    {   
       initCamshift(frame, selection, data);
    }
    if( DETECTION_MODE == CAMSHIFT_MODE )
    {
        camshiftTrack(frame, data.trackWindow, data);
        initHandDetection(frame, data.trackWindow, data);
        DETECTION_MODE = HANDDETECT_MODE;
    }
    else if( DETECTION_MODE == HANDDETECT_MODE )
    {
        camshiftTrack(frame, data.trackWindow, data);
    }
    // If face or hand has been lost, go back to FACEDETECT_MODE
    if( data.trackWindow.area() <= 1 )
    {
        data.isRunning = false;
        DETECTION_MODE = FACEDETECT_MODE;
    }
  
    Mat backprojectionFrame;
    resize(data.backproj, backprojectionFrame, Size( cvRound(data.backproj.cols/5), cvRound(data.backproj.rows/5)) , 0, 0, INTER_LINEAR);
    cv::imshow( "backproj", backprojectionFrame );     
}

void camshiftTrack( Mat& frame, Rect selection, camshiftData& data )
{
    cvtColor(frame, data.hsv, COLOR_BGR2HSV);

    inRange(data.hsv, Scalar(0, data.smin, MIN(data.vmin,data.vmax)), Scalar(180, 256, MAX(data.vmin, data.vmax)), data.mask);
    data.hue.create(data.hsv.size(), data.hsv.depth());
    mixChannels(&data.hsv, 1, &data.hue, 1, data.ch, 1); 
    // New backprojection
    calcBackProject(&data.hue, 1, 0, data.hist, data.backproj, &data.phueranges);
    data.backproj &= data.mask;
    RotatedRect trackBox = CamShift (data.backproj, data.trackWindow , TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ) );

    // Drawings
    rectangle( frame, data.trackWindow, CV_RGB(230,230,230), 3, 8, 0);
    rectangle( frame, data.trackWindow, CV_RGB(50,50,50), 1, 8, 0);   
    ellipse( frame, trackBox, Scalar(100,100,255), 3, CV_AA );     
}

void initHandDetection( Mat& frame, Rect selection, camshiftData& data )
{
   // To detect hand skin, we first remove head skin
    // 5% expand for better removal
    int begX = data.trackWindow.x - (frame.cols/20) ;
    int endX = data.trackWindow.x + data.trackWindow.width + (frame.rows/20);
    
    if( begX < 0 ) 
    {
        begX = 0;
    }
    if( endX > frame.cols )
    {
        endX = frame.cols;
    } 

   // Erase all the height (neck, hair,E...)
    for( int y = 0 ; y < frame.rows ; y++ )
    {
        for( int x = begX ; x < endX; x++ )
        {
            data.backproj.at<uchar>(y,x) = 0;
        }
    }

    // Threshold backproj
    for( int y = 0 ; y < data.backproj.rows ; y++ )
    {
        for( int x = 0 ; x < data.backproj.cols ; x++ )
        {
            if( data.backproj.at<uchar>(y,x) <= CAMSHIFT_BACKPROJ_THRESHOLD )
            {
                data.backproj.at<uchar>(y,x) = 0;
            }
            else
            {
                data.backproj.at<uchar>(y,x) = 255;
            }
        }
    }

    imwrite( "./data/backproj_image.jpg", data.backproj );
    // Moving trackWindow to right of face : where the left hand should be
    data.trackWindow.x += data.trackWindow.width;
    data.trackWindow.y = 0;
    data.trackWindow.width = data.backproj.cols - 1 - data.trackWindow.x;
    data.trackWindow.height = data.backproj.rows - 1;
    RotatedRect trackBox = CamShift(data.backproj, data.trackWindow , TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ) );    
}