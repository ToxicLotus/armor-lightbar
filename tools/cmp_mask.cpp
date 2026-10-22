#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 生成掩码对比图：左=原图 | 中=方案A(R-G>60,gray>100,3x3) | 右=方案B(R-G>20,gray>20,2x2)
// 用法: ./cmp <视频> <帧号> [帧号...]
// 不给帧号时默认 0 20 40 60 80 100

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

// 放大 3 倍，方便看细节；四角画个角标方便区分
static cv::Mat zoom3(const cv::Mat& m, int frameIdx, const char* label) {
    cv::Mat bgr;
    cv::cvtColor(m, bgr, cv::COLOR_GRAY2BGR);
    cv::Mat z;
    cv::resize(bgr, z, cv::Size(), 3.0, 3.0, cv::INTER_NEAREST);
    cv::putText(z, std::string(label) + " f" + std::to_string(frameIdx),
                cv::Point(8, 26), 0, 0.7, cv::Scalar(0, 255, 0), 2);
    return z;
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

        cv::Mat A = maskA(frame);
        cv::Mat B = maskB(frame);

        // 三列统一缩放到同一高度（hconcat 要求高度相同）
        const int H = 720;
        cv::Mat canvas, a3, b3;
        cv::resize(frame, canvas, cv::Size(), (double)H / frame.rows, (double)H / frame.rows, cv::INTER_AREA);

        cv::Mat Abgr, Bbgr;
        cv::cvtColor(A, Abgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(B, Bbgr, cv::COLOR_GRAY2BGR);
        cv::resize(Abgr, a3, cv::Size(), (double)H / A.rows, (double)H / A.rows, cv::INTER_NEAREST);
        cv::resize(Bbgr, b3, cv::Size(), (double)H / B.rows, (double)H / B.rows, cv::INTER_NEAREST);

        cv::putText(canvas, "original f" + std::to_string(f), cv::Point(8, 30),
                    0, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(a3, "A: R-G>60 gray>100 3x3", cv::Point(8, 30),
                    0, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(b3, "B: R-G>20 gray>20 2x2", cv::Point(8, 30),
                    0, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::Mat row;
        cv::hconcat(canvas, a3, row);
        cv::hconcat(row,    b3, row);

        std::string out = "../output/02_调试排查/mask_cmp_f" + std::to_string(f) + ".png";
        if (cv::imwrite(out, row)) std::cout << "saved " << out
            << "   A px=" << cv::countNonZero(A)
            << "  B px=" << cv::countNonZero(B) << "\n";
        else std::cerr << "save failed " << out << "\n";
    }
    return 0;
}
