#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 掩码对比：左=原图，中=方案A(R-G>60,gray>100,3x3)，右=方案B'(R-G>20,gray>50,2x2)
// 用法: ./cmp_mask <视频> [帧1 帧2 ...]
// 不给帧号时默认 0 20 40 60 80 100

static cv::Mat makeMask(const cv::Mat& f, int rgT, int gT, int ksize) {
    std::vector<cv::Mat> ch;
    cv::split(f, ch);
    cv::Mat d, mc;
    cv::subtract(ch[2], ch[1], d);
    cv::threshold(d, mc, rgT, 255, cv::THRESH_BINARY);

    cv::Mat g, mb;
    cv::cvtColor(f, g, cv::COLOR_BGR2GRAY);
    cv::threshold(g, mb, gT, 255, cv::THRESH_BINARY);

    cv::Mat m;
    cv::bitwise_and(mc, mb, m);
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::morphologyEx(m, m, cv::MORPH_OPEN, k);
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " <video> [frame...]\n"; return 2; }
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
    const int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::vector<int> frames;
    if (argc > 2) { for (int i = 2; i < argc; ++i) frames.push_back(std::atoi(argv[i])); }
    else frames = {0, 20, 40, 60, 80, 100};

    for (int f : frames) {
        if (f < 0 || f >= total) { std::cerr << "frame " << f << " out of range\n"; continue; }
        cap.set(cv::CAP_PROP_POS_FRAMES, f);
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) { std::cerr << "read fail " << f << "\n"; continue; }

        cv::Mat A = makeMask(frame, 60, 100, 3);    // 第4课原参数
        cv::Mat B = makeMask(frame, 20,  50, 2);    // 新参数：亮度提到 50

        const int H = 720;
        cv::Mat canvas, a3, b3;
        double s = (double)H / frame.rows;
        cv::resize(frame, canvas, cv::Size(), s, s, cv::INTER_AREA);

        cv::Mat Abgr, Bbgr;
        cv::cvtColor(A, Abgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(B, Bbgr, cv::COLOR_GRAY2BGR);
        cv::resize(Abgr, a3, cv::Size(), s, s, cv::INTER_NEAREST);
        cv::resize(Bbgr, b3, cv::Size(), s, s, cv::INTER_NEAREST);

        cv::putText(canvas, "original f" + std::to_string(f), cv::Point(8, 30), 0, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(a3, "A: RG>60 gray>100 3x3", cv::Point(8, 30), 0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(b3, "B: RG>20 gray>50 2x2", cv::Point(8, 30), 0, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::Mat row;
        cv::hconcat(canvas, a3, row);
        cv::hconcat(row, b3, row);

        std::string out = "../output/02_调试排查/cmp2_f" + std::to_string(f) + ".png";
        if (cv::imwrite(out, row)) {
            std::cout << "saved " << out
                      << "   A=" << cv::countNonZero(A)
                      << "  B=" << cv::countNonZero(B) << "\n";
        } else {
            std::cerr << "save failed " << out << "\n";
        }
    }
    return 0;
}
