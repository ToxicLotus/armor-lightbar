#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

struct Bar {
    cv::RotatedRect r;
    double a = 0;
    float L = 0, S = 0;
};

static double na(float x) {
    double v = x;
    while (v <= -90) v += 180;
    while (v > 90) v -= 180;
    return v;
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return 1;
    cv::Mat f;
    int idx = 0;
    while (cap.read(f)) {
        std::vector<cv::Mat> ch;
        cv::split(f, ch);
        cv::Mat d, mc;
        cv::subtract(ch[2], ch[1], d);
        cv::threshold(d, mc, 20, 255, cv::THRESH_BINARY);
        cv::Mat g, mb;
        cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
        cv::threshold(g, mb, 50, 255, cv::THRESH_BINARY);
        cv::Mat m;
        cv::bitwise_and(mc, mb, m);
        cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(m, m, cv::MORPH_OPEN, k);

        std::vector<std::vector<cv::Point>> cs;
        std::vector<cv::Vec4i> hh;
        cv::findContours(m.clone(), cs, hh, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Bar> v;
        for (size_t i = 0; i < cs.size(); ++i) {
            Bar b;
            b.a = cv::contourArea(cs[i]);
            if (b.a < 10) continue;
            b.r = cv::minAreaRect(cs[i]);
            b.L = std::max(b.r.size.width, b.r.size.height);
            b.S = std::min(b.r.size.width, b.r.size.height);
            if (b.S < 1e-6f) continue;
            if (b.L / b.S < 1.5) continue;
            if (b.L < 12) continue;
            v.push_back(b);
        }
        std::sort(v.begin(), v.end(), [](const Bar& x, const Bar& y) { return x.a > y.a; });

        if (idx % 10 == 0 || idx < 3) {
            printf("f%-3d n=%zu ", idx, v.size());
            for (size_t i = 0; i < v.size() && i < 3; ++i)
                printf("| #%zu a=%.0f cx=%.1f cy=%.1f L=%.1f W=%.1f ang=%.0f ",
                       i, v[i].a, v[i].r.center.x, v[i].r.center.y, v[i].L, v[i].S, na(v[i].r.angle));
            printf("\n");
        }
        ++idx;
    }
    return 0;
}
