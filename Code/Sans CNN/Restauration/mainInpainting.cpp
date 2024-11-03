#include <opencv2/opencv.hpp>
#include <opencv2/xphoto/inpainting.hpp>
#include <iostream>
 
using namespace cv;
 
int main(int argc, char** argv)
{
    Mat original_, mask_;
    original_ = imread("../Assets/test.jpg");
    mask_ = imread("../Assets/Mask.png", IMREAD_GRAYSCALE);
 
    Mat mask;
    resize(mask_, mask, original_.size(), 0.0, 0.0, cv::INTER_NEAREST);
 
    Mat im_distorted(original_.size(), original_.type(), Scalar::all(0));
    original_.copyTo(im_distorted, mask);
 
    Mat reconstructed;
    xphoto::inpaint(im_distorted, mask, reconstructed, xphoto::INPAINT_FSR_FAST);
 
    imshow("orignal image", original_);
    imshow("distorted image", im_distorted);
    imshow("reconstructed image", reconstructed);
    waitKey();
 
    return 0;
}