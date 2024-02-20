#include "yolovpose.h"
#include "opencv2/imgproc.hpp"
#include <iostream>

YolovPose::YolovPose(const std::string &model, const cv::Size size, float mScorThres, float mNMSThres)
{
    this->model_ = cv::dnn::readNetFromONNX(model);
    this->model_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->model_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    this->mShape_ = size;
    this->mScoreThreshold_ = mScorThres;
    this->mNMSThreshold_ = mNMSThres;

}

YolovPose::~YolovPose()
{

}

void YolovPose::initModel(const std::string &modelPath)
{
    this->model_ = cv::dnn::readNetFromONNX(modelPath);
    this->model_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->model_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

}

std::vector<BoundBoxe> YolovPose::detect(cv::Mat &img, int w, int h)
{
    static cv::Mat blob;
    static std::vector<cv::Mat> outputs;

    float dx = (w * 1.0) / (mShape_.width * 1.0);
    float dy = (h* 1.0) / (mShape_.height * 1.0);

    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, mShape_, cv::Scalar(), true, false);
    model_.setInput(blob);
    auto tick = cv::TickMeter();
    tick.start();
    model_.forward(outputs, model_.getUnconnectedOutLayersNames());
    tick.stop();
    std::clog << tick.getFPS() << "FPS works\n";
    const int channels = outputs[0].size[2];
    const int anchors = outputs[0].size[1];
    outputs[0] = outputs[0].reshape(1, anchors);
    cv::Mat output = outputs[0].t();

    std::vector<cv::Rect> bboxList;
    std::vector<float> scoreList;
    std::vector<int> indicesList;
    std::vector<std::vector<Keypoint>> kpList;

    for (int i = 0; i < channels; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto bbox_ptr = row_ptr;
        auto score_ptr = row_ptr + 4;
        auto kp_ptr = row_ptr + 5;

        float score = *score_ptr;
        if (score > mScoreThreshold_) {
            float x = *bbox_ptr++;
            float y = *bbox_ptr++;
            float w = *bbox_ptr++;
            float h = *bbox_ptr;

            float x0 = std::clamp((x - 0.5f * w) * 1.0F, 0.f, float(mShape_.width)) ;
            float y0 = std::clamp((y - 0.5f * h) * 1.0F, 0.f, float(mShape_.height)) ;
            float x1 = std::clamp((x + 0.5f * w) * 1.0F, 0.f, float(mShape_.width)) ;
            float y1 = std::clamp((y + 0.5f * h) * 1.0F, 0.f, float(mShape_.height)) ;

            x0 *= dx;
            y0 *= dy;

            x1 *=dx;
            y1 *= dy;

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;

            bbox.width = (x1 - x0) ;
            bbox.height = (y1 - y0) ;

            std::vector<Keypoint> kps;
            for (int k = 0; k < YolovPose::keyPointCount; k++) {
                float kps_x = (*(kp_ptr + 3 * k));
                float kps_y = (*(kp_ptr + 3 * k + 1)) ;
                float kps_s = *(kp_ptr + 3 * k + 2);
                kps_x = std::clamp(kps_x, 0.f, float(mShape_.width));
                kps_y = std::clamp(kps_y, 0.f, float(mShape_.height));

                kps.emplace_back(std::round(kps_x*dx), std::round(kps_y*dy), kps_s);
            }

            bboxList.push_back(bbox);
            scoreList.push_back(score);
            kpList.push_back(kps);
        }
    }

    cv::dnn::NMSBoxes(
        bboxList,
        scoreList,
        mScoreThreshold_,
        mNMSThreshold_,
        indicesList
        );

    std::vector<BoundBoxe> result{};
    for (auto &i: indicesList) {
        result.emplace_back(bboxList[i], scoreList[i], kpList[i]);
    }

    return result;
}

int YolovPose::getModelWidth() const
{
    return mShape_.width;
}

int YolovPose::getModelheight() const
{
    return mShape_.height;
}

cv::Mat YolovPose::preprocces(cv::Mat &img)
{
    cv::Mat res;
    cv::resize(img,res,mShape_);
    return res;
}

