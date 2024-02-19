#include "boundboxe.h"

BoundBoxe::BoundBoxe() {}

BoundBoxe::~BoundBoxe()
{
    this->kps_.clear();
}

cv::Rect2i BoundBoxe::getBoxe() const
{
    return this->boxe_;
}

std::vector<Keypoint> BoundBoxe::getKeyPoints() const
{
    return this->kps_;
}

uint8_t BoundBoxe::getKeyPointCount() const
{
    return static_cast<std::uint8_t>(kps_.size());
}

Keypoint &BoundBoxe::operator[](uint8_t index)
{
    return this->kps_[index];
}

Keypoint BoundBoxe::operator[](std::uint8_t index)const
{
    return this->kps_[index];
}
