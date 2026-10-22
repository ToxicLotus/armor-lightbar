#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

struct Bar {
    double area = 0, aspect = 0;
    float longSide = 0, shortSide = 0;
};

static std::vector<Bar> findBars(const cv::Mat& mask, double minLong) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask.clone(), contours, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Bar> bars;
    for (size_t i = 0; i < contours.size(); ++i) {
        Bar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 30) continue;
        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        b.longSide  = std::max(rr.size.width, rr.size.height);
        b.shortSide = std::min(rr.size.width, rr.size.height);
        if (b.shortSide < 1e-6f) continue;
        b.aspect = b.longSide / b.shortSide;
        if (b.aspect < 2.0) continue;
        if (b.longSide < minLong) continue;
        bars.push_back(b);
    }
    return bars;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    int want = (argc > 2) ? std::atoi(argv[2]) : 60;

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }
    cap.set(cv::CAP_PROP_POS_FRAMES, want);
    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) { std::cerr << "cannot read frame\n"; return 1; }

    std::vector<cv::Mat> ch;
    cv::split(frame, ch);
    cv::Mat diff, gray;
    cv::subtract(ch[2], ch[1], diff);
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 更低的阈值范围
    const int rgs[] = {10, 15, 20, 30};
    const int gs[]  = {10, 20, 30, 40, 60};

    std::cout << "frame " << want << "   (rows=RG threshold, cols=gray threshold)\n";
    std::cout << "RG\\gray|";
    for (int g : gs) printf("  g%3d", g);
    std::cout << "\n--------+";
    for (size_t i = 0; i < sizeof(gs)/sizeof(gs[0]); ++i) printf("------");
    std::cout << "\n";

    for (int rg : rgs) {
        printf("  RG%3d |", rg);
        for (int g : gs) {
            cv::Mat mc, mb, mask;
            cv::threshold(diff, mc, rg, 255, cv::THRESH_BINARY);
            cv::threshold(gray, mb, g,  255, cv::THRESH_BINARY);
            cv::bitwise_and(mc, mb, mask);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

            std::vector<Bar> bars = findBars(mask, 15);
            printf(" %d/%4d", (int)bars.size(), cv::countNonZero(mask));
        }
        printf("\n");
    }

    // 把几组掩码存成图片，肉眼确认"包得全不全"
    for (int rg : {10, 20, 30}) {
        for (int g : {20, 30, 40}) {
            cv::Mat mc, mb, mask;
            cv::threshold(diff, mc, rg, 255, cv::THRESH_BINARY);
            cv::threshold(gray, mb, g,  255, cv::THRESH_BINARY);
            cv::bitwise_and(mc, mb, mask);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

            std::string out = "../output/mask_rg" + std::to_string(rg)
                            + "_g" + std::to_string(g) + ".png";
            cv::imwrite(out, mask);
        }
    }
    std::cout << "\nmasks saved to ../output/mask_rg*_g*.png\n";
    return 0;
}