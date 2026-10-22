#include <iostream>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// 生成掩码对比视频
//   默认布局（三栏）：[原视频] [方案A掩码] [方案B掩码]
//   可选布局：--two   只出两版：A视频 / B视频（分别保存）
// 用法: ./maskvideo <输入视频> [--two]
// 输出: output/03_最终结果/mask_compare.avi
//       （--two 时输出 mask_A.avi 和 mask_B.avi）

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <video> [--two]\n";
        return 2;
    }
    std::string path = argv[1];
    bool two = (argc > 2 && std::string(argv[2]) == "--two");

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open: " << path << "\n"; return 1; }

    const int    w     = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    h     = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps   = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double useFps = (fps > 0 ? fps : 25.0);

    std::cout << "video: " << path << "\n";
    std::cout << "size : " << w << "x" << h << "  fps=" << fps
              << "  frames=" << total << "\n";

    // MJPG + .avi 在 Linux 上最稳
    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer;
    cv::VideoWriter wa, wb;

    std::string out = "../output/03_最终结果/mask_compare.avi";
    if (two) {
        if (!wa.open("../output/03_最终结果/mask_A.avi", fourcc, useFps, cv::Size(w * 2, h))) {
            std::cerr << "cannot open mask_A.avi\n"; return 1;
        }
        if (!wb.open("../output/03_最终结果/mask_B.avi", fourcc, useFps, cv::Size(w * 2, h))) {
            std::cerr << "cannot open mask_B.avi\n"; return 1;
        }
        std::cout << "saving: ../output/03_最终结果/mask_A.avi  (left=original, right=A mask)\n";
        std::cout << "saving: ../output/03_最终结果/mask_B.avi  (left=original, right=B mask)\n";
    } else {
        if (!writer.open(out, fourcc, useFps, cv::Size(w * 3, h))) {
            std::cerr << "cannot open " << out << "\n"; return 1;
        }
        std::cout << "saving: " << out << "  (left=original, mid=A mask, right=B mask)\n";
    }

    double scale = w / 640.0;
    cv::Mat frame;
    int idx = 0;

    while (cap.read(frame)) {
        cv::Mat A = maskA(frame);
        cv::Mat B = maskB(frame);

        cv::Mat Abgr, Bbgr;
        cv::cvtColor(A, Abgr, cv::COLOR_GRAY2BGR);
        cv::cvtColor(B, Bbgr, cv::COLOR_GRAY2BGR);

        cv::Mat orig = frame.clone();
        cv::putText(orig, "original " + std::to_string(idx),
                    cv::Point(10, (int)(30 * scale)), 0, 0.7 * scale, cv::Scalar(0, 255, 255), 2);

        if (two) {
            cv::Mat rowA, rowB;
            cv::hconcat(orig, Abgr, rowA);
            cv::hconcat(orig, Bbgr, rowB);
            wa.write(rowA);
            wb.write(rowB);
        } else {
            cv::putText(Abgr, "A: R-G>60 gray>100 3x3", cv::Point(10, (int)(30 * scale)),
                        0, 0.7 * scale, cv::Scalar(0, 255, 0), 2);
            cv::putText(Bbgr, "B: R-G>20 gray>20 2x2", cv::Point(10, (int)(30 * scale)),
                        0, 0.7 * scale, cv::Scalar(0, 255, 0), 2);
            cv::Mat row;
            cv::hconcat(orig, Abgr, row);
            cv::hconcat(row, Bbgr, row);
            writer.write(row);
            cv::imshow("mask compare", row);          // 同时实时显示
            char key = (char)cv::waitKey(1);
            if (key == 27) break;
        }
        ++idx;
    }

    std::cout << "processed " << idx << " / " << total << " frames\n";
    cap.release();
    if (two) { wa.release(); wb.release(); }
    else     { writer.release(); cv::destroyAllWindows(); }
    return 0;
}
