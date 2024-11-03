#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("../Assets/Test2.jpg");
    cv::Mat grayImage;

    if (image.empty()) {
        std::cerr << "Impossible de charger l'image." << std::endl;
        return -1;
    }

    // RGB --> NDG
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    int K;
    std::cout << "Choisir un NB de K : ";
    std::cin >> K;

    if (K < 2) {
        std::cerr << "K > 2 obligatoirement" << std::endl;
        return -1;
    }

    cv::Mat data;
    grayImage.convertTo(data, CV_32F);
    data = data.reshape(1, grayImage.rows * grayImage.cols);

    cv::Mat labels, centres;

    // K-Means
    cv::kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1.0), 10, cv::KMEANS_RANDOM_CENTERS, centres);

    cv::Mat colorImage(grayImage.size(), CV_8UC3);

    std::vector<cv::Vec3f> clusterColors(K, cv::Vec3f(0, 0, 0));
    std::vector<int> clusterCounts(K, 0);

    // Accumulation des couleurs / cluster
    for (int i = 0; i < grayImage.rows; ++i) {
        for (int j = 0; j < grayImage.cols; ++j) {
            int clusterIdx = labels.at<int>(i * grayImage.cols + j);
            cv::Vec3b color = image.at<cv::Vec3b>(i, j);
            clusterColors[clusterIdx] += color;
            clusterCounts[clusterIdx]++;
        }
    }

    // Calcul de la couleur moyenne / cluster
    for (int i = 0; i < K; ++i) {
        if (clusterCounts[i] > 0) {
            clusterColors[i] /= clusterCounts[i];
        }
    }

    // Création de l'image colorisée à l'aide des couleurs moyennes
    for (int i = 0; i < grayImage.rows; ++i) {
        for (int j = 0; j < grayImage.cols; ++j) {
            int clusterIdx = labels.at<int>(i * grayImage.cols + j);
            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<uchar>(clusterColors[clusterIdx][0]),
                static_cast<uchar>(clusterColors[clusterIdx][1]),
                static_cast<uchar>(clusterColors[clusterIdx][2])
            );
        }
    }

    cv::imshow("Image originale", image);
    cv::imshow("Image en NDG", grayImage);
    cv::imshow("Image colorisée", colorImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
