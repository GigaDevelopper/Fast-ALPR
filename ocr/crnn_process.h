#ifndef CRNN_PROCESS_H
#define CRNN_PROCESS_H

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/imgproc.hpp>

cv::Mat CrnnResizeImg(cv::Mat img, float wh_ratio, int rec_image_height);

std::vector<std::string> ReadDict(std::string path);

cv::Mat GetRotateCropImage(cv::Mat srcimage, std::vector<std::vector<int>> box);

template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

#endif // CRNN_PROCESS_H
