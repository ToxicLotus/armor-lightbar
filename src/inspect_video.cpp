#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Tool: print video properties and guess whether the light bars are red or blue.
// Usage: ./inspect_video <video path> [frame index]
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <video path> [frame index]" << std::endl;
        return 2;
    }
    std::string path = argv[1];
    int wantFrame = (argc > 2) ? std::atoi(argv[2]) : 0;

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << path << std::endl;
        return 1;
    }

    const int    w     = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    h     = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps   = cap.get(cv::CAP_PROP_FPS);
    const int    total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "video    : " << path << std::endl;
    std::cout << "resolution: " << w << " x " << h << std::endl;
    std::cout << "fps       : " << fps << std::endl;
    std::cout << "frames    : " << total << std::endl;
    if (fps > 0) std::cout << "duration  : " << total / fps << " s" << std::endl;

    if (wantFrame > 0) cap.set(cv::CAP_PROP_POS_FRAMES, wantFrame);

    cv::Mat frame;
    if (!cap.read(frame) || frame.empty()) {
        std::cerr << "cannot read frame " << wantFrame << std::endl;
        return 1;
    }

    std::vector<cv::Mat> ch;
    cv::split(frame, ch);                 // ch[0]=B ch[1]=G ch[2]=R
    double meanB = cv::mean(ch[0])[0];
    double meanG = cv::mean(ch[1])[0];
    double meanR = cv::mean(ch[2])[0];
    std::cout << "\nframe " << wantFrame << " mean: B=" << meanB
              << " G=" << meanG << " R=" << meanR << std::endl;
    std::cout << "guess: light bars are "
              << (meanR >= meanB ? "RED" : "BLUE") << std::endl;

    std::cout << "press any key to close the window..." << std::endl;
    cv::imshow("first frame", frame);
    cv::waitKey(0);
    return 0;
}
