#include "cls_procces.h"


#include <vector>

const std::vector<int> rec_image_shape{3, 48, 192};

cv::Mat ClsResizeImg(cv::Mat img) {
    int imgC, imgH, imgW;
    imgC = rec_image_shape[0];
    imgH = rec_image_shape[1];
    imgW = rec_image_shape[2];

    float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);

    int resize_w, resize_h;
    if (ceilf(imgH * ratio) > imgW)
        resize_w = imgW;
    else
        resize_w = int(ceilf(imgH * ratio));
    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
               cv::INTER_LINEAR);
    if (resize_w < imgW) {
        cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0, imgW - resize_w,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    return resize_img;
}
