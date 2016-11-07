#include "constants.h"

void detectAndDraw(Mat& frame, CascadeClassifier& cascade, bool tryflip, Rect& searchRect)
{
    double timer = 0; 
    vector<Rect> foundFaces, foundFaces2;
    Mat smallImg( cvRound(searchRect.height), cvRound(searchRect.width), CV_8UC1 );
    Mat grayFrame; // We'll work on a grayFramescale frame

    // Cropped zone is converted to grayFramescale
    cvtColor( frame(searchRect).clone(), grayFrame, CV_BGR2GRAY ); 
    // We resize it to fit smallImg size
    resize( grayFrame, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    timer = (double)cvGetTickCount();

    // Face detection via haar cascades
   	cascade.detectMultiScale( smallImg, foundFaces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, foundFaces2, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30) );

       if( !foundFaces2.empty() )
       {
       	    Rect faceRect = foundFaces2[0];
       	    foundFaces.push_back(Rect(smallImg.cols - faceRect.x - faceRect.width, faceRect.y, faceRect.width, faceRect.height));
       }
    }

    timer = (double)cvGetTickCount() - timer;
    printf( "detection time = %g ms\n", timer/((double)cvGetTickFrequency()*1000.) );

    if( foundFaces.empty() ) // If no faces found
    {
    	// searchRect go back to full frame
        searchRect = Rect(0, 0, frame.cols, frame.rows);
    }

    // Otherwise, searchRect is updated for an area around detected face
    else
    {
        Rect faceRect = foundFaces[0];
        Point center;
        int radius;    
    	// local searchRect oordinates are converted to full frame coordinates
        center.x = cvRound( searchRect.x + faceRect.x + faceRect.width * 0.5 );
        center.y = cvRound( searchRect.y + faceRect.y + faceRect.height * 0.5 );
        radius = cvRound( (faceRect.width + faceRect.height) * 0.25);
    
        // Widen search rect around detected face
        {
            int x = searchRect.x + faceRect.x - (int) FACEDETECT_SEARCH_RECT_OFFSET;
            int y = searchRect.y + faceRect.y - (int) FACEDETECT_SEARCH_RECT_OFFSET;
            int width = faceRect.width + 2 * (int) FACEDETECT_SEARCH_RECT_OFFSET;
            int height = faceRect.height + 2 * (int) FACEDETECT_SEARCH_RECT_OFFSET;
            int x2 = x + width;
            int y2 = y + height;
            // Checks to fit rect inside frame boundaries
            if( x < 0 ) 
                {
                    x = 0;
                }
            if( y < 0 )
            {
                y = 0;
            } 
            if( x2 > frame.cols )
                {
                    width = frame.cols - x;
                }
            if( y2 > frame.rows ) 
                {
                     height = frame.rows - y;
                }

            searchRect = Rect(x ,y , width, height);
        }	    
    }

    rectangle( frame, searchRect, CV_RGB(204,0,102), 2, 8, 0);
    cv::imshow( "result", frame );        
}