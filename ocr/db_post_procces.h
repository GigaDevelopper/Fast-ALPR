#ifndef DB_POST_PROCCES_H
#define DB_POST_PROCCES_H


#include <map>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

template <class T> T clamp(T x, T min, T max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

void GetContourArea(std::vector<std::vector<float>> box, float unclip_ratio,
                    float &distance);

cv::RotatedRect Unclip(std::vector<std::vector<float>> box, float unclip_ratio);

std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

bool XsortFp32(std::vector<float> a, std::vector<float> b);

bool XsortInt(std::vector<int> a, std::vector<int> b);

std::vector<std::vector<int>>
OrderPointsClockwise(std::vector<std::vector<int>> pts);

std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, float &ssid);

float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

std::vector<std::vector<std::vector<int>>>
BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                std::map<std::string, double> Config);

std::vector<std::vector<std::vector<int>>>
FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h,
                float ratio_w, cv::Mat srcimg);

#endif // DB_POST_PROCCES_H
