#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

struct Blob {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    double aspect = 0;
};

static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);
    cv::threshold(diff, maskColor, 20, 255, cv::THRESH_BINARY);

    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, 20, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);

    // 轻量开运算去掉 1 像素噪点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }
    const int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::Mat frame;
    int idx = 0;
    int n0 = 0, n1 = 0, n2 = 0, nMore = 0;

    printf("%-6s %-4s %-28s %-28s %s\n", "frame", "n", "bar0(area long short)", "bar1(area long short)", "separated?");
    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> h;
        cv::findContours(mask.clone(), contours, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Blob> blobs;
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area < 10) continue;
            Blob b;
            b.area = area;
            b.rect = cv::minAreaRect(contours[i]);
            b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
            b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
            b.aspect = (b.shortSide > 1e-6f) ? b.longSide / b.shortSide : 0.0;
            if (b.aspect < 1.5) continue;      // 只要细长的
            if (b.longSide < 12) continue;
            blobs.push_back(b);
        }
        std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) { return a.area > b.area; });

        int n = (int)blobs.size();
        if (n == 0) ++n0; else if (n == 1) ++n1; else if (n == 2) ++n2; else ++nMore;

        if (idx % 10 == 0 || idx >= total - 3) {
            char b0[64] = "-", b1[64] = "-";
            if (n >= 1) snprintf(b0, sizeof b0, "%6.0f %5.1f %5.1f", blobs[0].area, blobs[0].longSide, blobs[0].shortSide);
            if (n >= 2) snprintf(b1, sizeof b1, "%6.0f %5.1f %5.1f", blobs[1].area, blobs[1].longSide, blobs[1].shortSide);
            bool sep = (n >= 2) ? (blobs[0].area > blobs[1].area * 1.3) : false;
            printf("%-6d %-4d %-28s %-28s %s\n", idx, n, b0, b1, (n >= 2 ? (sep ? "yes(likely only 2 real)" : "NO") : "-"));
        }
        ++idx;
    }
    printf("\ntotal frames %d  -> blobs==0:%d  ==1:%d  ==2:%d  >2:%d\n", idx, n0, n1, n2, nMore);
    return 0;
}
