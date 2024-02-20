#ifndef YOLOVPOSE_H
#define YOLOVPOSE_H

#include "boundboxe.h"

#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/dnn.hpp>

#include <string>

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
    YolovPose(const YolovPose &) = default;
    YolovPose(YolovPose &&) = delete;
    YolovPose &operator=(const YolovPose &) = delete;
    YolovPose &operator=(YolovPose &&) = delete;
    explicit YolovPose(const std::string &model,
                       const cv::Size size = cv::Size(640, 640),
                       float mScorThres = 0.8, float mNMSThres = 0.7);
    ~YolovPose();

    void initModel(const std::string &modelPath);
    std::vector<BoundBoxe> detect(cv::Mat &img, int width = 640, int h = 320);
    const int keyPointCount{4};

    int getModelWidth() const;
    int getModelheight() const;

    cv::Mat preprocces(cv::Mat &img);


private:
    cv::dnn::Net model_;
    cv::Size mShape_;
    float mScoreThreshold_;
    float mNMSThreshold_;

};

#endif // YOLOVPOSE_H
