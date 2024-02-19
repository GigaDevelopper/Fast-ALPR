#ifndef YOLOVPOSE_H
#define YOLOVPOSE_H

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/dnn.hpp>


/*

YOLOV 8 Format
{
    'names': ['person'],
    'boxes': tensor([[x1, y1, x2, y2, conf, cls_idx]]),
    'kp': tensor([[x1_kpt_0, y1_kpt_0, score_0], ... [x1_kpt_n, y1_kpt_n, score_n]])
}
 */

class YolovPose
{
public:
    YolovPose();

private:
    cv::dnn::Net model_;
    cv::Size mShape_;
    float mScoreThreshold_;
    float mNMSThreshold_;

};

#endif // YOLOVPOSE_H
