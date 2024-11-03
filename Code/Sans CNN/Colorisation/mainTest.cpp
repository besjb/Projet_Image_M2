#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void ComputeColorMappingRGB(const cv::Mat& refImg, std::vector<cv::Vec3b>& colorMapping) {
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    std::vector<cv::Mat> bgrPlanes;
    cv::split(refImg, bgrPlanes);

    cv::Mat bHist, gHist, rHist;
    cv::calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), bHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), gHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), rHist, 1, &histSize, &histRange);

    cv::normalize(bHist, bHist, 0, 255, cv::NORM_MINMAX);
    cv::normalize(gHist, gHist, 0, 255, cv::NORM_MINMAX);
    cv::normalize(rHist, rHist, 0, 255, cv::NORM_MINMAX);

    colorMapping.resize(256);
    for (int i = 0; i < 256; ++i) {
        uchar b = static_cast<uchar>(bHist.at<float>(i));
        uchar g = static_cast<uchar>(gHist.at<float>(i));
        uchar r = static_cast<uchar>(rHist.at<float>(i));
        colorMapping[i] = cv::Vec3b(b, g, r);
    }
}

void ColorisationRGB(const cv::Mat& grayImage, const std::vector<cv::Vec3b>& colorMapping, cv::Mat& colorImage) {
    colorImage = cv::Mat::zeros(grayImage.size(), CV_8UC3);

    for (int y = 0; y < grayImage.rows; ++y) {
        for (int x = 0; x < grayImage.cols; ++x) {
            uchar grayValue = grayImage.at<uchar>(y, x);
            colorImage.at<cv::Vec3b>(y, x) = colorMapping[grayValue];
        }
    }
}

int main() {
    cv::Mat image = cv::imread("../Assets/ExpoUniv.jpg");
    cv::Mat imageRef = cv::imread("../Assets/Test4.jpg");
    cv::Mat grayImage, colorImage;

    if (image.empty() || imageRef.empty()) {
        std::cerr << "Impossible de charger l'une des images." << std::endl;
        return -1;
    }

    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::vector<cv::Vec3b> colorMapping;
    ComputeColorMappingRGB(imageRef, colorMapping);

    ColorisationRGB(grayImage, colorMapping, colorImage);

    cv::imshow("Image originale", image);
    cv::imshow("Image en NDG", grayImage);
    cv::imshow("Image colorisée", colorImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
