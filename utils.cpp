void drawText(cv::Mat& frame, string text, int fontSize, int xpos, int ypos, CvScalar color)
{
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = fontSize;
    int thickness = 3;
    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;
    Point textOrg(xpos  - textSize.width/2, ypos + textSize.height);
    putText(frame, text, textOrg, fontFace, fontScale, color, thickness, 8);
}