#include <iostream>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

// 从一个视频里抓指定帧号存成图片
// 用法: ./grab <视频路径> <帧号1> [帧号2] [帧号3] ...
// 例:   ./grab ../video/装甲板视频.avi 0 10 50

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <video> <frame1> [frame2] ..." << std::endl;
        return 2;
    }

    const std::string path = argv[1];
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << path << std::endl;
        return 1;
    }

    const int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "video: " << path << "  total frames: " << total << std::endl;

    for (int i = 2; i < argc; ++i) {
        int idx = std::atoi(argv[i]);
        if (idx < 0 || idx >= total) {
            std::cerr << "skip frame " << idx << " (out of range 0.." << total - 1 << ")" << std::endl;
            continue;
        }

        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "cannot read frame " << idx << std::endl;
            continue;
        }

        std::string out = "../output/grab_" + std::to_string(idx) + ".png";
        if (cv::imwrite(out, frame)) {
            std::cout << "saved " << out << std::endl;
        } else {
            std::cerr << "save failed: " << out << std::endl;
        }
    }
    return 0;
}