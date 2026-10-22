#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// 最终方案验证：
//   掩码用 R-G 和 gray 两级阈值，扫几组参数
//   对每组参数统计：找到 2 根候选的帧数 / 候选里"前二能配成一对"的帧数

struct Bar {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
};

static cv::Mat makeMask(const cv::Mat& f, int rgT, int gT) {
    std::vector<cv::Mat> ch; cv::split(f, ch);
    cv::Mat d, mc; cv::subtract(ch[2], ch[1], d);
    cv::threshold(d, mc, rgT, 255, cv::THRESH_BINARY);
    cv::Mat g, mb; cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
    cv::threshold(g, mb, gT, 255, cv::THRESH_BINARY);
    cv::Mat m; cv::bitwise_and(mc, mb, m);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
    return m;
}

static double normAngle(float a) {
    double v = a;
    while (v <= -90.0) v += 180.0;
    while (v >   90.0) v -= 180.0;
    return v;
}

// 前二是否能配成一对
static bool twoFormAPair(const std::vector<Bar>& bars) {
    if (bars.size() < 2) return false;
    const Bar& a = bars[0];
    const Bar& b = bars[1];
    double dy = std::fabs(a.rect.center.y - b.rect.center.y);
    double hAvg = 0.5 * (a.rect.size.height + b.rect.size.height);
    double dx = std::fabs(a.rect.center.x - b.rect.center.x);
    double areaRatio = b.area / a.area;
    double dAng = std::fabs(normAngle(a.rect.angle) - normAngle(b.rect.angle));
    return dy < std::max(10.0, 0.20 * hAvg) && areaRatio > 0.5 && dx > 5.0 && dAng < 30.0;
}

static std::vector<Bar> candidates(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> cs;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask.clone(), cs, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Bar> out;
    for (size_t i = 0; i < cs.size(); ++i) {
        Bar b;
        b.area = cv::contourArea(cs[i]);
        if (b.area < 10) continue;
        b.rect = cv::minAreaRect(cs[i]);
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;
        if (b.longSide / b.shortSide < 1.5) continue;
        if (b.longSide < 12) continue;
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(), [](const Bar& x, const Bar& y) { return x.area > y.area; });
    return out;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";

    struct Param { int rg, g; };
    Param params[] = {{20,20},{20,30},{20,40},{20,50},{30,30},{30,40},{30,50},{40,40}};

    for (const auto& p : params) {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }
        cv::Mat frame;
        int idx = 0, has2 = 0, isPair = 0, extra = 0;
        while (cap.read(frame)) {
            std::vector<Bar> c = candidates(makeMask(frame, p.rg, p.g));
            if (c.size() >= 2) { ++has2; if (twoFormAPair(c)) ++isPair; }
            if (c.size() > 2) ++extra;
            ++idx;
        }
        printf("RG=%2d gray=%2d : frames=%d  has>=2:%3d  top2-pair:%3d  (frames with >2 cands: %d)\n",
               p.rg, p.g, idx, has2, isPair, extra);
    }
    return 0;
}
