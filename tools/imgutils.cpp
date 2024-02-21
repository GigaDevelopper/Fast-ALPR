#include "imgutils.h"

#include <vector>
#include <cmath>

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
    int width = static_cast<int>(std::sqrt(std::pow(points[0].x - points[1].x,2)) +
                                 std::sqrt(std::pow(points[0].y - points[1].y,2)));

    int height =static_cast<int>(std::sqrt(std::pow(points[0].x - points[3].x,2)) +
                                  std::sqrt(std::pow(points[0].y - points[3].y,2)));

    cv::Point2f srcPoint[] = {
        points[0],
        points[1],
        points[2],
        points[3]

    };
    width = std::max(width,300);
    height = std::max(120,height);

    cv::Point2f dstPoint[] = {
        cv::Point2f(0.,0.),
        cv::Point2f(width,0),
        cv::Point2f(width,height),
        cv::Point2f(0,height),
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
        auto it = ImgUtils::getRotateCropImage(image, kp);
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

cv::Mat ImgUtils::getRotateCropImage(cv::Mat srcimage, std::vector<cv::Point2f>& box)
{
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<cv::Point2f> points = box;

    float x_collect[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    float y_collect[4] = {box[0].y, box[1].y, box[2].y, box[3].y};

    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }

    int img_crop_width =
        static_cast<int>(sqrt(pow(points[0].x - points[1].x, 2) +
                              pow(points[0].y - points[1].y, 2)));
    int img_crop_height =
        static_cast<int>(sqrt(pow(points[0].x - points[3].x, 2) +
                              pow(points[0].y - points[3].y, 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0].x, points[0].y);
    pointsf[1] = cv::Point2f(points[1].x, points[1].y);
    pointsf[2] = cv::Point2f(points[2].x, points[2].y);
    pointsf[3] = cv::Point2f(points[3].x, points[3].y);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    const float ratio = 4.5;
    if (static_cast<float>(dst_img.rows) >=
        static_cast<float>(dst_img.cols) * ratio) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}
