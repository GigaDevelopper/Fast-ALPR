#ifndef BOUNDBOXE_H
#define BOUNDBOXE_H

#include "keypoints.h"

#include <opencv4/opencv2/core.hpp>

#include <vector>

class BoundBoxe
{
public:
    BoundBoxe();
    explicit BoundBoxe(cv::Rect2i _box, float score, std::vector<Keypoint>&kp);
    BoundBoxe(const BoundBoxe &other);
    BoundBoxe(BoundBoxe &&other);
    ~BoundBoxe();

    BoundBoxe &operator = (const BoundBoxe &other);
    BoundBoxe &operator = (BoundBoxe &&other) noexcept;

    cv::Rect2i getBoxe()const;
    std::vector<Keypoint> getKeyPoints()const;
    std::uint8_t getKeyPointCount()const;
    float getConfidence()const;
    Keypoint &operator[](std::uint8_t index);
    Keypoint operator[](std::uint8_t index)const;

private:
    cv::Rect2i boxe_{};
    float conf_;
    std::vector<Keypoint>kps_{};

};

#endif // BOUNDBOXE_H
