#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 对比两种掩码方案在整条视频上的表现：
//   方案 A：第4课原始参数  R-G>60 且 gray>100，核 3x3   （掩码干净，但灯条只剩芯）
//   方案 B：现在用的参数  R-G>20 且 gray>20 ，核 2x2，取面积前二（掩码含反光）

struct Blob {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    double aspect = 0;
};

static cv::Mat maskA(const cv::Mat& f) {
    std::vector<cv::Mat> ch; cv::split(f, ch);
    cv::Mat d, mc; cv::subtract(ch[2], ch[1], d);
    cv::threshold(d, mc, 60, 255, cv::THRESH_BINARY);
    cv::Mat g, mb; cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
    cv::threshold(g, mb, 100, 255, cv::THRESH_BINARY);
    cv::Mat m; cv::bitwise_and(mc, mb, m);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
    return m;
}

static cv::Mat maskB(const cv::Mat& f) {
    std::vector<cv::Mat> ch; cv::split(f, ch);
    cv::Mat d, mc; cv::subtract(ch[2], ch[1], d);
    cv::threshold(d, mc, 20, 255, cv::THRESH_BINARY);
    cv::Mat g, mb; cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
    cv::threshold(g, mb, 20, 255, cv::THRESH_BINARY);
    cv::Mat m; cv::bitwise_and(mc, mb, m);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
    return m;
}

static std::vector<Blob> blobsOf(const cv::Mat& mask, double minAspect, float minLong) {
    std::vector<std::vector<cv::Point>> cs;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask.clone(), cs, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Blob> out;
    for (size_t i = 0; i < cs.size(); ++i) {
        Blob b;
        b.area = cv::contourArea(cs[i]);
        if (b.area < 10) continue;
        b.rect = cv::minAreaRect(cs[i]);
        b.longSide = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;
        b.aspect = b.longSide / b.shortSide;
        if (b.aspect < minAspect) continue;
        if (b.longSide < minLong) continue;
        out.push_back(b);
    }
    std::sort(out.begin(), out.end(), [](const Blob& a, const Blob& b){ return a.area > b.area; });
    return out;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }

    cv::Mat frame;
    int idx = 0;
    int a0 = 0, a1 = 0, a2 = 0, amore = 0;
    int b0 = 0, b1 = 0, b2 = 0, bmore = 0;
    double sumLenA = 0, sumLenB = 0; int cntA = 0, cntB = 0;
    std::vector<int> badA, badB;

    while (cap.read(frame)) {
        cv::Mat mA = maskA(frame);
        cv::Mat mB = maskB(frame);

        std::vector<Blob> ba = blobsOf(mA, 2.0, 25);
        std::vector<Blob> bb = blobsOf(mB, 1.5, 12);

        int na = std::min((int)ba.size(), 2);
        int nb = std::min((int)bb.size(), 2);
        if (ba.size() == 0) ++a0; else if (ba.size() == 1) ++a1; else if (ba.size() == 2) ++a2; else ++amore;
        if (bb.size() == 0) ++b0; else if (bb.size() == 1) ++b1; else if (bb.size() == 2) ++b2; else ++bmore;

        if (na == 2) { sumLenA += ba[0].longSide + ba[1].longSide; ++cntA; } else badA.push_back(idx);
        if (nb == 2) { sumLenB += bb[0].longSide + bb[1].longSide; ++cntB; } else badB.push_back(idx);

        ++idx;
    }

    printf("frames: %d\n", idx);
    printf("A (R-G>60, gray>100, 3x3):  n=0:%d  n=1:%d  n=2:%d  n>2:%d | avg bar long=%.1f\n",
           a0, a1, a2, amore, cntA ? sumLenA / cntA : 0.0);
    printf("B (R-G>20, gray>20, 2x2) :  n=0:%d  n=1:%d  n=2:%d  n>2:%d | avg bar long=%.1f\n",
           b0, b1, b2, bmore, cntB ? sumLenB / cntB : 0.0);
    printf("A frames without 2 bars:");
    for (size_t i = 0; i < badA.size() && i < 20; ++i) printf(" %d", badA[i]);
    printf("%s\n", badA.size() > 20 ? " ..." : "");
    printf("B frames without 2 bars:");
    for (size_t i = 0; i < badB.size() && i < 20; ++i) printf(" %d", badB[i]);
    printf("%s\n", badB.size() > 20 ? " ..." : "");
    return 0;
}
