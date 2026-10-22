#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// 逐帧体检：
//   1. 候选对象数量
//   2. 画出的四边形中心 y，与"该帧最亮的红色区域"中心 y 的偏差
//      如果四边形跑到别的地方去了，偏差会很大
//   3. 输出最可疑的帧

struct Bar {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    cv::Point2f top, bottom;
};

static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff, mc;
    cv::subtract(ch[2], ch[1], diff);
    cv::threshold(diff, mc, 20, 255, cv::THRESH_BINARY);
    cv::Mat gray, mb;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mb, 50, 255, cv::THRESH_BINARY);
    cv::Mat m;
    cv::bitwise_and(mc, mb, m);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
    return m;
}

static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return 1;

    cv::Mat frame;
    int idx = 0;
    int bad = 0;
    // 统计：每帧候选数量分布
    int hist[8] = {0};

    printf("%-5s %-3s %-8s %-22s %-22s %s\n", "frm", "n", "areas", "bars y-range", "brightest-red y", "verdict");
    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<std::vector<cv::Point>> cs;
        std::vector<cv::Vec4i> h;
        cv::findContours(mask.clone(), cs, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Bar> cand;
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
            cand.push_back(b);
        }
        std::sort(cand.begin(), cand.end(), [](const Bar& a, const Bar& b) { return a.area > b.area; });

        int n = (int)cand.size();
        if (n > 7) n = 7;
        ++hist[n];

        // "最亮的红色区域" = 用严格亮度阈值找出来的灯条，作为参照
        std::vector<cv::Mat> ch;
        cv::split(frame, ch);
        cv::Mat diff, mc;
        cv::subtract(ch[2], ch[1], diff);
        cv::threshold(diff, mc, 20, 255, cv::THRESH_BINARY);
        cv::Mat gray, mb;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, mb, 120, 255, cv::THRESH_BINARY);   // 很高，只留真正的灯条
        cv::Mat ref;
        cv::bitwise_and(mc, mb, ref);
        std::vector<cv::Point> nz;
        cv::findNonZero(ref, nz);
        double refY = -1;
        if (!nz.empty()) {
            double s = 0;
            for (const auto& p : nz) s += p.y;
            refY = s / nz.size();
        }

        double barsY = -1;
        if (cand.size() >= 2) {
            getEndpoints(cand[0].rect, cand[0].top, cand[0].bottom);
            getEndpoints(cand[1].rect, cand[1].top, cand[1].bottom);
            barsY = 0.25 * (cand[0].top.y + cand[0].bottom.y + cand[1].top.y + cand[1].bottom.y);
        }

        bool suspect = (refY >= 0 && barsY >= 0 && std::fabs(barsY - refY) > 25);
        if (suspect) ++bad;

        if (idx % 5 == 0 || suspect) {
            printf("%-5d %-3d %-8.0f %-22.1f %-22.1f %s\n", idx, (int)cand.size(),
                   cand.empty() ? 0.0 : cand[0].area, barsY, refY, suspect ? "SUSPECT" : "ok");
        }
        ++idx;
    }
    printf("\ncandidate-count histogram (n -> frames):");
    for (int i = 0; i < 8; ++i) if (hist[i]) printf("  %d->%d", i, hist[i]);
    printf("\nsuspect frames: %d / %d\n", bad, idx);
    return 0;
}
