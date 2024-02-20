#ifndef IMGUTILS_H
#define IMGUTILS_H

#include "yolov8/boundboxe.h"

#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/photo.hpp>

class ImgUtils {
public :
    //show image
    static void show(cv::Mat &image);

    //read image
    static cv::Mat imageFromPath(const std::string &imagePath);

    //draw points and boxes
    static void draw(std::vector<BoundBoxe> &detections, cv::Mat &image);

    //crop region
    static cv::Mat cropAndTransform(const cv::Mat &image, const std::vector<cv::Point2f> &points);

    static cv::Mat cropByBoundBoxes(const cv::Mat &image, const std::vector<BoundBoxe> &boxes);

    static cv::Mat deskew(cv::Mat image);
};


#endif // IMGUTILS_H
