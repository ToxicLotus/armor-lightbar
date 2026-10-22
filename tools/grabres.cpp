#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

// 从任意视频抓帧存成 png，用于检查结果视频
int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: " << argv[0] << " <video> <frame> [frame...]\n"; return 2; }
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
    int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "video " << argv[1] << "  " << w << "x" << h << "  frames=" << total << "\n";

    for (int i = 2; i < argc; ++i) {
        int f = std::atoi(argv[i]);
        if (f < 0 || f >= total) { std::cerr << "frame " << f << " out of range\n"; continue; }
        cap.set(cv::CAP_PROP_POS_FRAMES, f);
        cv::Mat fr;
        if (!cap.read(fr) || fr.empty()) { std::cerr << "read fail " << f << "\n"; continue; }
        std::string out = "../output/02_调试排查/resvid_" + std::to_string(f) + ".png";
        if (cv::imwrite(out, fr)) std::cout << "saved " << out << "  (" << fr.cols << "x" << fr.rows << ")\n";
    }
    return 0;
}
