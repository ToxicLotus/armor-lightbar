#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// ============================================================================
// 任务二：装甲板灯条识别
//
// 流程（对应任务书四步）：
//   ① 掩码 + 二值化 ：cv::threshold          -> 只留灯条像素
//   ② 识别灯条轮廓   ：cv::findContours       -> 提取候选
//   ③ 算上下端点     ：cv::minAreaRect + 几何 -> 每根灯条的上下端点
//   ④ 四端点连矩形   ：cv::line               -> 画出装甲板框
//
// 逐帧处理用 cv::VideoCapture，显示用 cv::imshow，另存结果视频。
//
// 【参数是怎么定的】对队伍视频 101 帧扫描：
//   R-G > 20 且 gray > 50  ->  每帧恰好剩 2 个候选，且就是那两根灯条
//   地面反光（暗红光）被 gray>50 滤掉，不需要额外的位置/大小规则
// ============================================================================

struct LightBar {
    cv::RotatedRect rect;
    double  area = 0;
    float   longSide = 0, shortSide = 0;
    cv::Point2f top, bottom;
};

// 角度归一化到 (-90, 90]
static double normAngle(float a) {
    double v = a;
    while (v <= -90.0) v += 180.0;
    while (v >   90.0) v -= 180.0;
    return v;
}

// ---- 第 ① 步：掩码 + 二值化 ----
static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    cv::split(frame, ch);                       // ch[0]=B ch[1]=G ch[2]=R

    // 颜色：灯条是红的 -> R 明显大于 G
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);
    cv::threshold(diff, maskColor, 20, 255, cv::THRESH_BINARY);

    // 亮度：灯条在发光，地面反光只是"暗红" -> 亮度能分开两者
    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, 50, 255, cv::THRESH_BINARY);

    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);

    // 开运算去孤立噪点。用 2x2 而不是 3x3：远处灯条只有 2~3 像素宽
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// 从旋转矩形算上下端点：四角按 y 排序，上两点中点 / 下两点中点
static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);                               // 4 个角点，顺序不保证
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

// ---- 第 ②③ 步：找轮廓 -> 筛候选 -> 取面积最大的两根 -> 算端点 ----
static std::vector<LightBar> findLightBars(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // findContours 会修改输入图，所以传 clone()
    cv::findContours(mask.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        LightBar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 10) continue;              // 噪点

        b.rect = cv::minAreaRect(contours[i]);
        // size.width 不保证是长边，必须 max/min
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;

        double aspect = b.longSide / b.shortSide;
        if (aspect < 1.5) continue;             // 不够细长
        if (b.longSide < 12) continue;          // 太短
        cand.push_back(b);
    }

    // 面积从大到小，只取前两名（就是两根灯条）
    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });

    // 自检：如果找到了两个以上的候选，前两名必须"像一对灯条"；
    // 不像就说明掩码里有别的东西在冒充，这时只保留最大的一个，宁可少画也不乱画。
    if (cand.size() >= 3) {
        const LightBar& a = cand[0];
        const LightBar& b = cand[1];
        double dy = std::fabs(a.rect.center.y - b.rect.center.y);
        double hAvg = 0.5 * (a.rect.size.height + b.rect.size.height);
        double areaRatio = b.area / a.area;
        double dAngle = std::fabs(normAngle(a.rect.angle) - normAngle(b.rect.angle));
        bool plausible = (dy < std::max(12.0, 0.25 * hAvg))
                      && (areaRatio > 0.35 && areaRatio < 2.8)
                      && (dAngle < 35.0);
        if (!plausible) cand.resize(1);
    }

    if (cand.size() > 2) cand.resize(2);

    for (size_t i = 0; i < cand.size(); ++i) {
        getEndpoints(cand[i].rect, cand[i].top, cand[i].bottom);
    }
    return cand;
}

int main(int argc, char** argv) {
    std::string videoPath = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    bool saveVideo = !(argc > 2 && std::string(argv[2]) == "--no-save");

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << videoPath << "\n";
        return -1;
    }

    const int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps    = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double useFps = (fps > 0.0 ? fps : 25.0);

    std::cout << "video : " << videoPath << "\n";
    std::cout << "size  : " << width << " x " << height
              << "   fps: " << fps << "   frames: " << total << "\n";

    // 结果视频：MJPG + .avi 在 Linux 上最稳
    cv::VideoWriter writer;
    std::string outPath = "../output/03_最终结果/result.avi";
    if (saveVideo) {
        writer.open(outPath, cv::VideoWriter::fourcc('M','J','P','G'), useFps,
                    cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "warning: cannot open " << outPath << ", will not save\n";
            saveVideo = false;
        }
    }

    double scale = width / 640.0;
    cv::Mat frame;
    int index = 0;
    int missFrames = 0;

    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<LightBar> bars = findLightBars(mask);

        // 按 x 分左右，保证连线是 左-右-右-左，不会交叉
        if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x)
            std::swap(bars[0], bars[1]);

        cv::Mat canvas = frame.clone();

        // 每根灯条自己的框 + 端点
        for (size_t i = 0; i < bars.size(); ++i) {
            cv::Point2f p[4];
            bars[i].rect.points(p);
            for (int k = 0; k < 4; ++k)
                cv::line(canvas, p[k], p[(k + 1) % 4], cv::Scalar(0, 255, 0), 1);
            cv::circle(canvas, bars[i].top,    3, cv::Scalar(0, 255, 255), -1);  // 上端点 黄
            cv::circle(canvas, bars[i].bottom, 3, cv::Scalar(255, 0, 255), -1);  // 下端点 紫
        }

        // ---- 第 ④ 步：四端点按顺序连线 ----
        if (bars.size() == 2) {
            const LightBar& L = bars[0];
            const LightBar& R = bars[1];
            cv::Point2f quad[4] = { L.top, R.top, R.bottom, L.bottom };
            for (int i = 0; i < 4; ++i)
                cv::line(canvas, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
        } else {
            ++missFrames;
        }

        cv::putText(canvas,
                    "frame " + std::to_string(index) + "   bars: " + std::to_string(bars.size()),
                    cv::Point(8, (int)(26 * scale)), 0, 0.7 * scale,
                    cv::Scalar(0, 255, 255), 2);

        // 右下角放一个小掩码，方便看它干不干净
        cv::Mat maskBgr, maskSmall;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
        cv::resize(maskBgr, maskSmall, cv::Size(width / 4, height / 4));
        maskSmall.copyTo(canvas(cv::Rect(width - maskSmall.cols, height - maskSmall.rows,
                                         maskSmall.cols, maskSmall.rows)));

        if (saveVideo) writer.write(canvas);

        cv::imshow("lightbar", canvas);
        char key = (char)cv::waitKey(30);
        if (key == 27) break;

        ++index;
    }

    std::cout << "processed " << index << " / " << total << " frames";
    if (missFrames) std::cout << "   (frames without full quad: " << missFrames << ")";
    std::cout << "\n";
    if (saveVideo) std::cout << "result video: " << outPath << "\n";

    cap.release();
    if (saveVideo) writer.release();
    cv::destroyAllWindows();
    return 0;
}
