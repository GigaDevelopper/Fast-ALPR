#include "imgutils.h"
#include <vector>

void ImgUtils::show(cv::Mat &image)
{
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cv::waitKeyEx();
}

cv::Mat ImgUtils::imageFromPath(const std::string &imagePath)
{
    return cv::imread(imagePath);
}

void ImgUtils::draw(std::vector<BoundBoxe> &detections, cv::Mat &image)
{
    auto textColor = cv::Scalar(255, 255, 255);
    auto boxColor = cv::Scalar(255,  0, 0);

    for (BoundBoxe &item: detections) {
        cv::rectangle(image, item.getBoxe(), boxColor, 1);

        std::string infoString = std::to_string(item.getConfidence());
        cv::Size textSize = cv::getTextSize(infoString, cv::QT_FONT_NORMAL, 1, 1, nullptr);
        cv::Rect textBox(item.getBoxe().x, item.getBoxe().y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(image, textBox, boxColor, cv::FILLED);
        cv::putText(image, infoString, cv::Point(item.getBoxe().x + 5, item.getBoxe().y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1,
                    0);
        for (Keypoint& kp:item.getKeyPoints()) {
            cv::circle(image, kp.getPosition(), 3, boxColor, cv::FILLED);
        }
    }
}

cv::Mat ImgUtils::cropAndTransform(const cv::Mat &image, const std::vector<cv::Point2f> &points)
{
    constexpr int width = 520;
    constexpr int height =220 ;

    cv::Point2f srcPoint[] = {
        points[0],
        points[1],
        points[3],
        points[2]

    };

    cv::Point2f dstPoint[] = {
        cv::Point2f(0,0),
        cv::Point2f(width,0),
        cv::Point2f(0,height),
        cv::Point2f(width,height)
    };
    cv::Mat Matrix =  cv::getPerspectiveTransform(srcPoint,dstPoint);
    cv::Mat out,res;
    cv::warpPerspective(image,out,Matrix,cv::Size(width, height));
    cv::fastNlMeansDenoisingColored(out,res,7,15,7,16);
    cv::Mat ans,gray,blur,binary;
    cv::convertScaleAbs(res,ans,2.5);
    cv::cvtColor(ans,gray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray,blur,cv::Size(7,7),0);
    cv::threshold(blur,binary,180,255,cv::THRESH_BINARY_INV|cv::THRESH_OTSU);
    return blur;
}

cv::Mat ImgUtils::cropByBoundBoxes(const cv::Mat &image, const std::vector<BoundBoxe> &boxes)
{
    for(const auto& bx:boxes)
    {
        std::vector<cv::Point2f> kp;
        for(const auto& kps:bx.getKeyPoints())
        {
            kp.push_back(cv::Point(kps.getX(),kps.getY()));
        }
        auto it = ImgUtils::cropAndTransform(image, kp);
        ImgUtils::show(it);
    }
    return image;
}

cv::Mat ImgUtils::deskew(cv::Mat image){

    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Get coordinates of non-zero pixels
    std::vector<cv::Point> co_ords;
    cv::findNonZero(gray, co_ords);

    // Get the minimum area rectangle
    cv::RotatedRect rect = cv::minAreaRect(co_ords);
    float angle = rect.angle;
    if (angle < -45) {
        angle = -(90 + angle);
    } else {
        angle = -angle;
    }

    cv::Size size = image.size();
    cv::Point2f center(size.width / 2, size.height / 2);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(image, rotated, M, size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    return rotated;
}
