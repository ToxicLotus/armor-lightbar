#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ============================================================================
// 第 6 课：选出两根灯条，并算出每根的上下端点
//
// 掩码里会出现 4 块东西：上排两根（真灯条），下排两根（地面反光）。
// 实测规律：真灯条的面积永远是最大的两个（差距 1.5~3 倍），
// 所以按面积排序取前两名，就是两根灯条。
//
// 端点怎么算：minAreaRect 给出的 4 个角点顺序不保证，
// 先按 y 排序，前两个是上面短边的两个角 -> 取中点 = 上端点，
// 后两个是下面短边的两个角 -> 取中点 = 下端点。
// ============================================================================

struct LightBar {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    double aspect = 0;
    cv::Point2f top, bottom;
};

// 第 4 课改进版掩码：阈值降到 20/20（真视频画面极暗），只做轻量开运算
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

    // 2x2 而不是 3x3：灯条只有 3~10 像素宽，3x3 会把它腐蚀掉
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// 算出灯条的上下端点
static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f pts[4];
    rr.points(pts);                        // 4 个角点，顺序不保证

    std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;                  // y 小的在上
    });

    top    = cv::Point2f((pts[0].x + pts[1].x) * 0.5f, (pts[0].y + pts[1].y) * 0.5f);
    bottom = cv::Point2f((pts[2].x + pts[3].x) * 0.5f, (pts[2].y + pts[3].y) * 0.5f);
}

// 找灯条：筛选几何特征 -> 按面积排序 -> 只留前两名
static std::vector<LightBar> findLightBars(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> h;
    cv::findContours(mask.clone(), contours, h,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 10) continue;

        LightBar b;
        b.area = area;
        b.rect = cv::minAreaRect(contours[i]);
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;
        b.aspect = b.longSide / b.shortSide;

        if (b.aspect < 1.5) continue;      // 不够细长
        if (b.longSide < 12) continue;     // 太短
        cand.push_back(b);
    }

    // 按面积从大到小，取前两名 = 两根真灯条
    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });
    if (cand.size() > 2) cand.resize(2);

    for (size_t i = 0; i < cand.size(); ++i) {
        getEndpoints(cand[i].rect, cand[i].top, cand[i].bottom);
    }
    return cand;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";

    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open video: " << path << "\n"; return -1; }

    const int    w     = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const double fps   = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double scale = w / 640.0;

    cv::Mat frame;
    int index = 0;

    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<LightBar> bars = findLightBars(mask);

        // 按 x 排序，确定左灯条 / 右灯条
        if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x) {
            std::swap(bars[0], bars[1]);
        }

        // 前 3 帧打印端点坐标
        if (bars.size() != 2) {
            std::cout << "frame " << index << " -> " << bars.size() << " bars" << std::endl;
        }

        // 画图
        cv::Mat canvas = frame.clone();
        for (size_t i = 0; i < bars.size(); ++i) {
            // 灯条自身的矩形（绿）
            cv::Point2f pts[4];
            bars[i].rect.points(pts);
            for (int k = 0; k < 4; ++k)
                cv::line(canvas, pts[k], pts[(k + 1) % 4], cv::Scalar(0, 255, 0), 2);
            // 端点：上=黄，下=紫
            cv::circle(canvas, bars[i].top,    4, cv::Scalar(0, 255, 255), -1);
            cv::circle(canvas, bars[i].bottom, 4, cv::Scalar(255, 0, 255), -1);
            // 灯条中轴（青）
            cv::line(canvas, bars[i].top, bars[i].bottom, cv::Scalar(255, 255, 0), 1);
        }

        cv::putText(canvas, "frame " + std::to_string(index) + " bars:" + std::to_string(bars.size()),
                    cv::Point(10, (int)(28 * scale)), 0, 0.7 * scale, cv::Scalar(0, 255, 255), 2);

        cv::Mat maskBgr;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
        cv::Mat view;
        cv::hconcat(canvas, maskBgr, view);

        if (index % 20 == 0) {
            std::string out = "../output/lesson6_frame_" + std::to_string(index) + ".png";
            if (!cv::imwrite(out, view)) std::cerr << "save failed: " << out << "\n";
        }

        cv::imshow("lesson 6", view);
        char key = (char)cv::waitKey(30);
        if (key == 27) break;
        ++index;
    }

    std::cout << "processed " << index << " / " << total << " frames" << std::endl;
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
