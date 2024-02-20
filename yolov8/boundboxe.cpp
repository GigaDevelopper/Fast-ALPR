#include "boundboxe.h"

BoundBoxe::BoundBoxe() {}

BoundBoxe::BoundBoxe(cv::Rect2i _box, float score, std::vector<Keypoint> &kp)
{
    this->boxe_ = _box;
    this->conf_ = score;
    this->kps_ = kp;
}

BoundBoxe::BoundBoxe(const BoundBoxe &other)
{
    this->conf_ = other.getConfidence();
    this->boxe_ = other.getBoxe();
    this->kps_ = other.getKeyPoints();
}

BoundBoxe::BoundBoxe(BoundBoxe &&other)
{
    this->conf_ = std::move(other.conf_);
    this->boxe_ = std::move(other.boxe_);
    this->kps_ = std::move(other.kps_);
}

BoundBoxe::~BoundBoxe()
{
    this->kps_.clear();
}

BoundBoxe &BoundBoxe::operator =(const BoundBoxe &other)
{
    if(this != &other)
    {
        this->conf_ = other.getConfidence();
        this->boxe_ = other.getBoxe();
        this->kps_ = other.getKeyPoints();
    }
    return *this;
}

BoundBoxe &BoundBoxe::operator =(BoundBoxe &&other) noexcept
{
    if(this != &other)
    {
        this->conf_ = std::move(other.conf_);
        this->boxe_ = std::move(other.boxe_);
        this->kps_ = std::move(other.kps_);
    }
    return *this;
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

float BoundBoxe::getConfidence() const
{
    return this->conf_;
}

Keypoint &BoundBoxe::operator[](uint8_t index)
{
    return this->kps_[index];
}

Keypoint BoundBoxe::operator[](std::uint8_t index)const
{
    return this->kps_[index];
}
