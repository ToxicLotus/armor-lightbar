#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// 跑一帧最终算法，把 原图+四边形 | 掩码 并排存成图片
// 用法: ./prev <video> <frame>

struct LightBar {
    cv::RotatedRect rect;
    double  area = 0;
    float   longSide = 0, shortSide = 0;
    cv::Point2f top, bottom;
};

static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);
    cv::threshold(diff, maskColor, 20, 255, cv::THRESH_BINARY);
    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, 50, 255, cv::THRESH_BINARY);
    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

static std::vector<LightBar> findLightBars(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        LightBar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 10) continue;
        b.rect = cv::minAreaRect(contours[i]);
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;
        if (b.longSide / b.shortSide < 1.5) continue;
        if (b.longSide < 12) continue;
        cand.push_back(b);
    }
    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });
    if (cand.size() > 2) cand.resize(2);
    for (size_t i = 0; i < cand.size(); ++i) getEndpoints(cand[i].rect, cand[i].top, cand[i].bottom);
    return cand;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: " << argv[0] << " <video> <frame>\n"; return 2; }
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }
    int want = std::atoi(argv[2]);

    cap.set(cv::CAP_PROP_POS_FRAMES, want);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) { std::cerr << "cannot read frame " << want << "\n"; return 1; }

    cv::Mat mask = makeMask(frame);
    std::vector<LightBar> bars = findLightBars(mask);
    if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x) std::swap(bars[0], bars[1]);

    cv::Mat canvas = frame.clone();
    for (size_t i = 0; i < bars.size(); ++i) {
        cv::Point2f p[4];
        bars[i].rect.points(p);
        for (int k = 0; k < 4; ++k) cv::line(canvas, p[k], p[(k + 1) % 4], cv::Scalar(0, 255, 0), 1);
        cv::circle(canvas, bars[i].top, 4, cv::Scalar(0, 255, 255), -1);
        cv::circle(canvas, bars[i].bottom, 4, cv::Scalar(255, 0, 255), -1);
    }
    if (bars.size() == 2) {
        cv::Point2f quad[4] = { bars[0].top, bars[1].top, bars[1].bottom, bars[0].bottom };
        for (int i = 0; i < 4; ++i) cv::line(canvas, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }
    cv::putText(canvas, "frame " + std::to_string(want) + " bars:" + std::to_string(bars.size()),
                cv::Point(10, 30), 0, 0.8, cv::Scalar(0, 255, 255), 2);
    for (size_t i = 0; i < bars.size(); ++i)
        std::cout << "  " << (i == 0 ? "LEFT " : "RIGHT")
                  << " top=(" << bars[i].top.x << "," << bars[i].top.y << ")"
                  << " bottom=(" << bars[i].bottom.x << "," << bars[i].bottom.y << ")\n";

    cv::Mat maskBgr;
    cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
    cv::Mat view;
    cv::hconcat(canvas, maskBgr, view);

    std::string out = "../output/02_调试排查/final_f" + std::to_string(want) + ".png";
    if (cv::imwrite(out, view)) std::cout << "saved " << out << "\n";
    else std::cerr << "save failed " << out << "\n";
    return 0;
}
