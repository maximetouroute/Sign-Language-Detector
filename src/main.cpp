/*
@Author Maxime Touroute
April 2015
*/

#include "unistd.h"
#include "signLanguageDetector.cpp"
#include "utils.cpp"

static void help()
{
    cout << "\nTP3, TP2 & TP1. Author: Maxime Touroute SI4 GMD\n"
    "During execution:\n\tHit any key to quit.\n"
    "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

int main( int argc, const char** argv )
{
    system("./scripts/init.sh");
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;

    // Loading HaarCascade, setting up basic parameters
    const string cascadeOpt = "--cascade=";
    size_t cascadeOptLen = cascadeOpt.length();
    const string tryFlipOpt = "--try-flip";
    size_t tryFlipOptLen = tryFlipOpt.length();
    string inputName;
    bool tryflip = false;

    DETECTION_MODE = FACEDETECT_MODE;
    PROGRAM_MODE = SIGN_TRAINING_MODE;
    camshiftData theCamshiftData;
    initCamshiftConstants(theCamshiftData);
    Rect searchRect;
    help();
    CascadeClassifier cascade;

    // Handling parameters & errors
    {
        for( int i = 1; i < argc; i++ )
        {
            if( tryFlipOpt.compare( 0, tryFlipOptLen, argv[i], tryFlipOptLen ) == 0 )
            {
                tryflip = true;
                cout << " will try to flip image horizontally to detect assymetric objects\n";
            }
            else if( argv[i][0] == '-' )
            {
                cerr << "WARNING: Unknown option %s" << argv[i] << endl;
            }
            else
            {
                inputName.assign( argv[i] );
            }
        }

        if( !cascade.load( cascadeName ) )
        {
            cerr << "ERROR: Could not load classifier cascade" << endl;
            help();
            return -1;
        }

        if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
        {
            capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );
            int c = inputName.empty() ? 0 : inputName.c_str()[0] - '0' ;
            if(!capture) cout << "Capture from CAM " <<  c << " didn't work" << endl;
        }  
    }

    cvNamedWindow( "result", 1 );
    if( capture )
    {
        cout << "In capture ..." << endl;

        // Search area is initialized at frame size
        {
            IplImage* iplImg = cvQueryFrame( capture );
            Mat temp = iplImg;
            searchRect = Rect(0, 0, temp.cols, temp.rows);
        }
        IplImage* iplImg;

        // Infinite loop on camera frames
        for(;;)
        {
            iplImg = cvQueryFrame( capture );
            frame = iplImg; 

            if( frame.empty() )
            {
                break;
            }
            if( iplImg->origin == IPL_ORIGIN_TL )
            {
                frame.copyTo( frameCopy );
            }
            else
            {
                flip( frame, frameCopy, 0 );
            }

            if( DETECTION_MODE == FACEDETECT_MODE )
            {
                detectAndDraw( frameCopy, cascade, tryflip, searchRect );
                // if search area got smaller than frame, it means a face has been found via haar detector
                // Thus, we switch to skinColor tracking (faster)
                if(searchRect.width < frame.cols) 
                {
                    DETECTION_MODE = CAMSHIFT_MODE;
                }
           }
           else if ( DETECTION_MODE == CAMSHIFT_MODE || DETECTION_MODE == HANDDETECT_MODE )
           {
            // If already tracking skinColor, We just keep tracking
            executeCamshift(frameCopy, searchRect, theCamshiftData); 
        }

        char keyboard_event = (char)waitKey(10);
        if ( keyboard_event == 27 )
        {
            goto _cleanup_;
        }
        // Reset detection
        if( keyboard_event == '\t' )
        {
            DETECTION_MODE = FACEDETECT_MODE;
            initCamshiftConstants(theCamshiftData);
        }
        // Change program mode
        if( keyboard_event == 32 )
        {
            if ( PROGRAM_MODE == SIGN_TRAINING_MODE )
            {
                system("./scripts/trainNetwork.sh");
                // Neural network data must exist
                if( access( "./data/neural/neuralData.xml", F_OK ) != -1 ) 
                {

                    PROGRAM_MODE = SIGN_RECOGNITION_MODE;
                } 
                else 
                {
                    Mat error_img( 200,500, CV_32FC3 );
                    string text = "ERROR ";
                    drawText(error_img, text, 2, error_img.cols/2, error_img.rows/2, CV_RGB(255,0,0));
                    text = "Neural file not found";
                    drawText(error_img, text, 2, error_img.cols/2, error_img.rows/2+30, CV_RGB(255,0,0));
                    cv::imshow( "Error", error_img );
                    cv::waitKey(3000);
                    cvDestroyWindow("Error");
                }
            }
            else 
            {
                PROGRAM_MODE = SIGN_TRAINING_MODE;
            }
        }

        if( PROGRAM_MODE == SIGN_TRAINING_MODE )
        {
            // If user clicked on an alphabet letter, draw letter on screen & compute training
            if ( keyboard_event >= 'a' && keyboard_event <= 'z')
            { 
                letterDisplayed = toupper(keyboard_event);
                train(frameCopy, theCamshiftData, keyboard_event);
            }
        }

        // Recognition only 1/10th frame for realtime purposes
        else if (PROGRAM_MODE == SIGN_RECOGNITION_MODE && frameCounter % 10 == 0)
        {    
        	recognize(frameCopy, theCamshiftData);
        }
        else 
        {}

    Mat drawedFrame = frameCopy;
        // Some text infos
    {
        string text;
        if( PROGRAM_MODE == SIGN_TRAINING_MODE )
        {
            text = "Training mode";
        }
        else
        {
            text = "Recognition mode";
        }
        drawText(drawedFrame, text, 4, drawedFrame.cols/2, 10, CV_RGB(204,100,20));
    }
    {
        string text;
        text.push_back(letterDisplayed);
        drawText(drawedFrame, text, 10, drawedFrame.cols-50, 50, CV_RGB(255,255,255));
    }
    {
        string text = "Change Mode [SPACE]";
        drawText(drawedFrame, text, 2, drawedFrame.cols/2, 60, CV_RGB(200,200,200));
        text = "Reset detection [TAB]";
        drawText(drawedFrame, text, 2, drawedFrame.cols/2, 90, CV_RGB(200,200,200));
    }

    cv::imshow( "result", drawedFrame );      
    frameCounter++;
    } // Camera loop

    waitKey(0);
    _cleanup_:
    cvReleaseCapture( &capture );
}

cvDestroyWindow("result");
return 0;
}