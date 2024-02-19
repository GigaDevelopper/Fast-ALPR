#ifndef KEYPOINTS_H
#define KEYPOINTS_H

//Represent keypoints in bounded Boxes
#include <opencv4/opencv2/core.hpp>

class Keypoint
{
public:
    Keypoint(){}
    explicit Keypoint(float x, float y, float score);
    Keypoint(const Keypoint &kp);
    Keypoint(Keypoint &&kp) noexcept;

    Keypoint &operator = (const Keypoint &kp);
    Keypoint &operator = (Keypoint &&kp) noexcept;

    ~Keypoint();

    float getX()const;
    float getY()const;
    const cv::Point2d& getPosition()const;
    float getConfidence()const;

    friend std::ostream &operator<<(std::ostream& out, const Keypoint& kp){
        out <<"Point2D("<<kp.getX()<<","<<kp.getY()<<")";
        return out;
    };

private:
    cv::Point2d pos_;//Position of points
    float conf_;//confidence
};

#endif // KEYPOINTS_H
