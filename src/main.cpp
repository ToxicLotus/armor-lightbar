#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 第 4 课：在颜色掩码之后，再加"亮度"和"形态学"两道筛子，
// 滤掉灯条的暗反光。

static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);                  // ch[0]=B ch[1]=G ch[2]=R

    // ---- 筛子 1：颜色（红）----
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);      // R - G
    cv::threshold(diff, maskColor, 60, 255, cv::THRESH_BINARY);

    // ---- 筛子 2：亮度 ----
    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, 100, 255, cv::THRESH_BINARY);

    // 两个条件必须同时满足：既红，又亮
    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);

    // ---- 筛子 3：形态学开运算，去掉剩下的小碎块 ----
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    //OPEN=开运算，开运算=先腐蚀再膨胀
    return mask;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << path << std::endl;
        return -1;
    }

    const int    w     = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    h     = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps   = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "video: " << path << std::endl;
    std::cout << "size : " << w << " x " << h
              << "   fps: " << fps << "   frames: " << total << std::endl;

    cv::Mat frame;
    int index = 0;
    int saved = 0;

    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);

        cv::Mat maskBgr;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);

        cv::Mat canvas;
        cv::hconcat(frame, maskBgr, canvas);

        cv::putText(canvas, "frame " + std::to_string(index),
                    cv::Point(10, 30), 0, 0.8, cv::Scalar(0, 255, 0), 2);

        if (index % 20 == 0) {
            std::string out = "../output/lesson4_frame_" + std::to_string(index) + ".png";
            if (cv::imwrite(out, canvas)) ++saved;
            else std::cerr << "save failed: " << out << std::endl;
        }

        cv::imshow("lesson 4", canvas);
        char key = (char)cv::waitKey(30);
        if (key == 27) break;

        ++index;
    }

    std::cout << "processed " << index << " frames, saved " << saved << " images"
              << std::endl;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}