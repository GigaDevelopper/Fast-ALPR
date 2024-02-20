#include "keypoints.h"

//Construct keyPoints
Keypoint::Keypoint(float x, float y, float score)
    :conf_{score}
{
    this->pos_ = cv::Point2d(x,y);
}

Keypoint::Keypoint(const Keypoint &kp)
{
    this->pos_ = kp.getPosition();
    this->conf_ = kp.getConfidence();
}

Keypoint::Keypoint(Keypoint &&kp) noexcept
{
    this->conf_ = std::move(kp.conf_);
    this->pos_ = std::move(kp.pos_);
}

Keypoint &Keypoint::operator =(const Keypoint &kp)
{
    if(this != &kp){
        this->pos_ = kp.getPosition();
        this->conf_ = kp.getConfidence();
    }
    return *this;
}

Keypoint & Keypoint::operator = (Keypoint &&kp) noexcept{

    if(this != &kp){
        this->conf_ = std::move(kp.conf_);
        this->pos_ = std::move(kp.pos_);
    }
    return *this;
}

Keypoint::~Keypoint()
{

}

float Keypoint::getX() const
{
    return this->pos_.x;
}

float Keypoint::getY() const
{
    return this->pos_.y;
}

const cv::Point2f &Keypoint::getPosition() const
{
    return this->pos_;
}

float Keypoint::getConfidence() const
{
    return conf_;
}



