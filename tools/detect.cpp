#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 侦探程序：把每一个红色连通块都检出来，按面积排序，
// 打印它们的尺寸，并把"面积最大的两个"用绿框标出来。

struct Blob {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    double aspect = 0;
    int id = 0;
};

static cv::Mat makeMask(const cv::Mat& frame, int rgThr, int grayThr) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);
    cv::threshold(diff, maskColor, rgThr, 255, cv::THRESH_BINARY);

    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, grayThr, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);
    return mask;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    int want = (argc > 2) ? std::atoi(argv[2]) : 60;

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }
    cap.set(cv::CAP_PROP_POS_FRAMES, want);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) { std::cerr << "cannot read\n"; return 1; }

    // 用之前扫描出来最稳的一组：RG=20, gray=20（先不做过强的形态学）
    cv::Mat mask = makeMask(frame, 20, 20);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask.clone(), contours, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Blob> blobs;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 10) continue;
        Blob b;
        b.area = area;
        b.id = (int)i;
        b.rect = cv::minAreaRect(contours[i]);
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        b.aspect = (b.shortSide > 1e-6f) ? b.longSide / b.shortSide : 0.0;
        blobs.push_back(b);
    }

    std::cout << "frame " << want << "  total blobs(area>=10): " << blobs.size() << "\n";
    // 按面积从大到小排序
    std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.area > b.area; });
    for (size_t i = 0; i < blobs.size(); ++i) {
        printf("  #%zu  area=%6.0f  long=%5.1f short=%5.1f aspect=%5.2f  center=(%5.1f,%5.1f) angle=%6.1f\n",
               i, blobs[i].area, blobs[i].longSide, blobs[i].shortSide, blobs[i].aspect,
               blobs[i].rect.center.x, blobs[i].rect.center.y, blobs[i].rect.angle);
    }

    // 画图：前两个（面积最大）用绿框，其余用红框
    cv::Mat canvas = frame.clone();
    for (size_t i = 0; i < blobs.size(); ++i) {
        cv::Point2f pts[4];
        blobs[i].rect.points(pts);
        cv::Scalar color = (i < 2) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        for (int k = 0; k < 4; ++k) {
            cv::line(canvas, pts[k], pts[(k + 1) % 4], color, 2);
        }
        cv::putText(canvas, std::to_string(i),
                    cv::Point((int)blobs[i].rect.center.x + 5, (int)blobs[i].rect.center.y),
                    0, 0.6, color, 2);
    }

    cv::Mat maskBgr;
    cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
    cv::Mat view;
    cv::hconcat(canvas, maskBgr, view);

    std::string out = "../output/detect_frame_" + std::to_string(want) + ".png";
    cv::imwrite(out, view);
    std::cout << "saved: " << out << "\n";
    return 0;
}
