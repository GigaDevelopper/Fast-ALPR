#include "tools/imgutils.h"
#include "yolov8/yolovpose.h"
using namespace std;
using namespace cv;;
int main()
{
    auto model = YolovPose("/home/azmiddin/Projects/Fast-ALPR/models/pose_platen.onnx",cv::Size(640,320),0.5,0.6);

    // cv::VideoCapture cap("/home/azmiddin/Projects/Fast-ALPR/test_example.mp4");
    // while (cap.isOpened()){
    //     cv::Mat frame;
    //     if (cap.read(frame)){
    //         cv::Mat image;
    //         image = model.preprocces(frame);
    //         auto result {model.detect(image, frame.cols, frame.rows)};
    //         ImgUtils::draw(result, frame);
    //         ImgUtils::cropByBoundBoxes(frame,result);
    //     }
    // }
    Mat frame = cv::imread("/home/azmiddin/Projects/Fast-ALPR/uzb_00128.jpg");
    ImgUtils::show(frame);
    if(!frame.empty())
    {           Mat image;
                image = model.preprocces(frame);
                auto result {model.detect(image, frame.cols, frame.rows)};
                ImgUtils::cropByBoundBoxes(frame,result);
                ImgUtils::draw(result, frame);
                ImgUtils::show(frame);
    }



    return 0;
}
