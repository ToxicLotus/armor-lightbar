#include <iostream>
#include <string>
#include <algorithm>             // 见下方注意事项
#include <vector>
#include <opencv2/opencv.hpp>

// 诊断用：打印掩码每一步还剩多少像素，找出到底哪一步把它们筛没了
static void diagnose(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);                      // ch[0]=B ch[1]=G ch[2]=R

    cv::Mat diff;
    cv::subtract(ch[2], ch[1], diff);
    double mn, mx;
    cv::minMaxLoc(diff, &mn, &mx);
    std::cout << "  R-G     min=" << mn << " max=" << mx
              << "  (if max < your threshold, color step is the problem)\n";

    cv::Mat maskColor;
    cv::threshold(diff, maskColor, 60, 255, cv::THRESH_BINARY);
    std::cout << "  after R-G>60   : " << cv::countNonZero(maskColor) << " px\n";

    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::minMaxLoc(gray, &mn, &mx);
    std::cout << "  gray    min=" << mn << " max=" << mx << "\n";

    cv::threshold(gray, maskBright, 100, 255, cv::THRESH_BINARY);
    std::cout << "  after gray>100 : " << cv::countNonZero(maskBright) << " px\n";

    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);
    std::cout << "  after AND      : " << cv::countNonZero(mask) << " px  <-- look here\n";

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat opened;
    cv::morphologyEx(mask, opened, cv::MORPH_OPEN, kernel);
    std::cout << "  after OPEN 3x3 : " << cv::countNonZero(opened) << " px\n";
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open: " << path << "\n"; return 1; }

    for (int want : {0, 30, 60, 100}) {
        cap.set(cv::CAP_PROP_POS_FRAMES, want);
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) continue;
        std::cout << "===== frame " << want << " =====\n";
        diagnose(frame);
    }
    return 0;
}