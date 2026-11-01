// ============================================================================
// 任务二：装甲板灯条识别 + 位姿解算
//   ① 颜色+亮度掩码             cv::threshold
//   ② 找灯条轮廓                cv::findContours
//   ③ 旋转矩形 -> 上下端点       cv::minAreaRect
//   ④ 四端点连成装甲板框         cv::line
//   ⑤ 位姿解算                  cv::solvePnP
//   ⑥ 画 FLU 三维坐标轴          cv::projectPoints
//
// 相机坐标系：x 右 / y 下 / z 前   （OpenCV 约定）
// 物体坐标系：x 前 / y 左 / z 上   （FLU）
// ============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

// 一根灯条：旋转矩形 + 面积 + 边长 + 上下端点
struct LightBar {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    cv::Point2f top, bottom;
};

// 角度归一化到 (-90, 90]
static double normAngle(float a) {
    double v = a;
    while (v <= -90.0) v += 180.0;
    while (v > 90.0) v -= 180.0;
    return v;
}

// 相机内参 K：[fx 0 cx; 0 fy cy; 0 0 1]，单位是像素
static cv::Mat cameraMatrix() {
    return (cv::Mat_<double>(3, 3) <<
        9.28130989e+02, 0.0,            3.77572945e+02,
        0.0,            9.30138391e+02, 2.83892859e+02,
        0.0,            0.0,            1.0);
}

// 畸变系数，顺序固定为 k1 k2 p1 p2 k3（切向夹在中间）
static cv::Mat distortionCoeffs() {
    return (cv::Mat_<double>(1, 5) <<
        -2.54433647e-01,  5.69431382e-01,
         3.65405229e-03, -1.09433818e-03, -1.33846840e+00);
}

// 装甲板 4 个端点的 3D 坐标（毫米）。装甲板是平面，所以 x 全为 0
static std::vector<cv::Point3f> armorObjectPoints() {
    const float L2 = 50.0f / 2.0f;    // 半灯条长（沿 z）
    const float W2 = 130.0f / 2.0f;   // 半灯条间距（沿 y）

    return {
        { 0.0f,  W2,  L2 },   // [0] 左灯条 上端点  y=+65(左) z=+25(上)
        { 0.0f, -W2,  L2 },   // [1] 右灯条 上端点
        { 0.0f, -W2, -L2 },   // [2] 右灯条 下端点
        { 0.0f,  W2, -L2 },   // [3] 左灯条 下端点
    };
}

// ① 掩码：红(R-G) 且 亮，开运算去孤立噪点
static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;                 // OpenCV 通道顺序是 BGR
    cv::split(frame, ch);

    cv::Mat diff, maskColor, gray, maskBright, mask;
    cv::subtract(ch[2], ch[1], diff);                       // R - G
    cv::threshold(diff, maskColor, 20, 255, cv::THRESH_BINARY);

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, maskBright, 50, 255, cv::THRESH_BINARY);

    cv::bitwise_and(maskColor, maskBright, mask);            // 两个条件同时满足

    // 用 2x2 而不是 3x3：远处灯条只有 2~3 像素宽，3x3 会把它整体腐蚀掉
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// ③ 旋转矩形 -> 长轴两端短边的中点（端点不是角点）
static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);                            // 返回顺序不保证，必须先按 y 排序
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

// ②③ 找轮廓 -> 按面积/长宽比/长度筛选 -> 取面积最大的两根
static std::vector<LightBar> findLightBars(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // findContours 会修改输入图，所以要传 clone
    cv::findContours(mask.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        LightBar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 10) continue;                       // 太小 = 噪点

        b.rect = cv::minAreaRect(contours[i]);
        // size.width 不保证是短边，必须用 max/min 求长宽比
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;               // 防除零

        if (b.longSide / b.shortSide < 1.5) continue;    // 不够细长
        if (b.longSide < 12) continue;                   // 太短
        cand.push_back(b);
    }

    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });

    // 自检：候选多于 2 个时，确认前二真像一对灯条，否则只保留最大的一个
    if (cand.size() >= 3) {
        const LightBar& a = cand[0];
        const LightBar& b = cand[1];
        double dy        = std::fabs(a.rect.center.y - b.rect.center.y);
        double hAvg      = 0.5 * (a.rect.size.height + b.rect.size.height);
        double areaRatio = b.area / a.area;
        double dAngle    = std::fabs(normAngle(a.rect.angle) - normAngle(b.rect.angle));
        bool plausible = (dy < std::max(12.0, 0.25 * hAvg))
                      && (areaRatio > 0.35 && areaRatio < 2.8)
                      && (dAngle < 35.0);
        if (!plausible) cand.resize(1);
    }

    if (cand.size() > 2) cand.resize(2);

    for (size_t i = 0; i < cand.size(); ++i)
        getEndpoints(cand[i].rect, cand[i].top, cand[i].bottom);
    return cand;
}

// ⑤ 位姿解算
static bool estimatePose(const std::vector<LightBar>& bars,
                         cv::Vec3d& rvec, cv::Vec3d& tvec) {
    if (bars.size() != 2) return false;

    // 2D 顺序必须和 armorObjectPoints() 一一对应，错了不报错但位姿会扭曲
    std::vector<cv::Point2f> imgPts = {
        bars[0].top, bars[1].top, bars[1].bottom, bars[0].bottom
    };

    // 4 个点共面，用 IPPE（专解平面目标）
    bool ok = cv::solvePnP(armorObjectPoints(), imgPts,
                           cameraMatrix(), distortionCoeffs(),
                           rvec, tvec,
                           false, cv::SOLVEPNP_IPPE);
    if (!ok) return false;                   // 失败时 rvec/tvec 是垃圾值

    // 平面 PnP 的朝向二义性：4 个 3D 点 x 全为 0，正反摆放投影完全一样。
    // 用先验"装甲板正面朝相机"选解：物体 x 轴在相机 z 上的分量 > 0 说明解成了背面。
    // 纠正 = 右乘 diag(-1,-1,1)，即绕物体 z 轴转 180°（行列式 +1 才是纯旋转）。
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    if (R.at<double>(2, 0) > 0.0) {
        cv::Mat flip = (cv::Mat_<double>(3, 3) <<
            -1.0,  0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0,  0.0, 1.0);
        R = R * flip;
        cv::Rodrigues(R, rvec);              // 必须转回 rvec，projectPoints 只接受 rvec
    }
    return true;
}

// ⑥ 画 FLU 坐标轴：projectPoints 是 solvePnP 的逆运算
static void drawAxes(cv::Mat& canvas, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                     float axisLen, int thickness) {
    // 物体坐标系下的原点 + 三个轴端点
    std::vector<cv::Point3f> axis3d = {
        { 0.0f,    0.0f,    0.0f    },
        { axisLen, 0.0f,    0.0f    },   // x 前
        { 0.0f,    axisLen, 0.0f    },   // y 左
        { 0.0f,    0.0f,    axisLen },   // z 上
    };

    std::vector<cv::Point2f> axis2d;
    cv::projectPoints(axis3d, rvec, tvec, cameraMatrix(), distortionCoeffs(), axis2d);

    // Scalar 是 BGR 顺序：x 红 / y 绿 / z 蓝
    const cv::Scalar color[3] = {
        cv::Scalar(0,   0, 255),
        cv::Scalar(0, 255,   0),
        cv::Scalar(255, 0,   0),
    };
    const char* name[3] = { "x", "y", "z" };

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    double r[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            r[a][b] = R.at<double>(a, b);

    // 分两轮画：先全部箭头再全部标签，否则后面的箭头会盖住前面的标签
    cv::Point o = axis2d[0];
    for (int i = 0; i < 3; ++i)
        cv::arrowedLine(canvas, o, axis2d[i + 1], color[i], thickness,
                        cv::LINE_AA, 0, 0.25);

    for (int i = 0; i < 3; ++i) {
        cv::Point tip = axis2d[i + 1];

        // P_cam = R * P_obj + t，手动展开矩阵乘法，最后换算成米
        const cv::Point3f& P = axis3d[i + 1];
        double cx = r[0][0]*P.x + r[0][1]*P.y + r[0][2]*P.z + tvec[0];
        double cy = r[1][0]*P.x + r[1][1]*P.y + r[1][2]*P.z + tvec[1];
        double cz = r[2][0]*P.x + r[2][1]*P.y + r[2][2]*P.z + tvec[2];

        char label[64];
        std::snprintf(label, sizeof label, "%s(%.1f,%.1f,%.1f)",
                      name[i], cx / 1000.0, cy / 1000.0, cz / 1000.0);

        // 沿"原点 -> 轴尖"方向外推 16 像素，避免三条轴的标签互相压住
        cv::Point d = tip - o;
        double dlen = std::sqrt((double)d.x * d.x + (double)d.y * d.y);
        cv::Point off = (dlen > 1.0)
                      ? cv::Point((int)(d.x / dlen * 16), (int)(d.y / dlen * 16))
                      : cv::Point(8, -8);

        const double fscale = 0.55;
        const int    fthick = 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, 0, fscale, fthick, &baseline);

        // putText 的坐标是文字左下角，所以按 左/右/上/下 四个方向夹到画面内
        cv::Point lpos = tip + off;
        if (lpos.x + ts.width > canvas.cols - 4) lpos.x = canvas.cols - 4 - ts.width;
        if (lpos.x < 4)                          lpos.x = 4;
        if (lpos.y - ts.height < 4)              lpos.y = 4 + ts.height;
        if (lpos.y > canvas.rows - 4)            lpos.y = canvas.rows - 4;

        cv::putText(canvas, label, lpos, 0, fscale, color[i], fthick);
    }
}

int main(int argc, char** argv) {
    std::string videoPath = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    bool saveVideo = !(argc > 2 && std::string(argv[2]) == "--no-save");

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {                   // 打开失败不抛异常，必须自己检查
        std::cerr << "cannot open video: " << videoPath << "\n";
        return -1;
    }

    const int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps    = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    const double useFps = (fps > 0.0 ? fps : 25.0);   // 有些视频 fps 返回 0

    std::cout << "video : " << videoPath << "\n";
    std::cout << "size  : " << width << " x " << height
              << "   fps: " << fps << "   frames: " << total << "\n";

    // 输出视频：MJPG + avi，帧率和尺寸必须和原视频一致
    cv::VideoWriter writer;
    std::string outPath = "../output/03_最终结果/result.avi";
    if (saveVideo) {
        writer.open(outPath, cv::VideoWriter::fourcc('M','J','P','G'), useFps,
                    cv::Size(width, height));
        if (!writer.isOpened()) {            // 失败也不该让整个程序挂掉
            std::cerr << "warning: cannot open " << outPath << ", will not save\n";
            saveVideo = false;
        }
    }

    double scale = width / 640.0;            // 字号按画面宽度缩放

    // 必须先 namedWindow(WINDOW_NORMAL) 才能 resizeWindow 放大窗口
    cv::namedWindow("lightbar", cv::WINDOW_NORMAL);
    cv::resizeWindow("lightbar", (int)(width * 1.8), (int)(height * 1.8));

    const float AXIS_LEN_MM = 200.0f;        // 只为好看，不影响解算结果

    cv::Mat frame;
    int index = 0, missFrames = 0, poseFrames = 0;

    while (cap.read(frame)) {                // read() 既取帧又返回是否成功
        cv::Mat mask = makeMask(frame);
        std::vector<LightBar> bars = findLightBars(mask);

        // 按 x 排成左、右，保证四边形连线不会交叉
        if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x)
            std::swap(bars[0], bars[1]);

        cv::Mat canvas = frame.clone();      // Mat 赋值是浅拷贝，必须 clone

        // 每根灯条：画旋转矩形 + 上下端点
        for (size_t i = 0; i < bars.size(); ++i) {
            cv::Point2f p[4];
            bars[i].rect.points(p);
            for (int k = 0; k < 4; ++k)
                cv::line(canvas, p[k], p[(k + 1) % 4], cv::Scalar(180, 180, 0), 1);
            cv::circle(canvas, bars[i].top,    3, cv::Scalar(0, 255, 255), -1);  // 上 黄
            cv::circle(canvas, bars[i].bottom, 3, cv::Scalar(255, 0, 255), -1);  // 下 紫
        }

        // ④ 四端点连矩形：左上 -> 右上 -> 右下 -> 左下
        if (bars.size() == 2) {
            cv::Point2f quad[4] = { bars[0].top,    bars[1].top,
                                    bars[1].bottom, bars[0].bottom };
            for (int i = 0; i < 4; ++i)
                cv::line(canvas, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        } else {
            ++missFrames;
        }

        // ⑤⑥ 位姿解算 + 画三维坐标轴
        cv::Vec3d rvec, tvec;
        if (estimatePose(bars, rvec, tvec)) {
            ++poseFrames;
            drawAxes(canvas, rvec, tvec, AXIS_LEN_MM, 2);

            if (index % 20 == 0) {           // 每 20 帧打印一次，避免刷屏
                std::cout << "frame " << index
                          << "  t = (" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ") mm"
                          << "  dist = " << cv::norm(tvec) / 1000.0 << " m\n";
            }
        }

        // 左上角显示帧号和检出数量（putText 的坐标是文字左下角）
        cv::putText(canvas,
                    "frame " + std::to_string(index) + "   bars: " + std::to_string(bars.size()),
                    cv::Point(8, (int)(26 * scale)), 0, 0.7 * scale,
                    cv::Scalar(0, 255, 255), 2);

        // 右下角嵌一个小掩码；ROI 与原图共享内存，copyTo 就等于贴图
        cv::Mat maskBgr, maskSmall;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);     // 通道数必须一致才能 copyTo
        cv::resize(maskBgr, maskSmall, cv::Size(width / 4, height / 4));
        maskSmall.copyTo(canvas(cv::Rect(width - maskSmall.cols, height - maskSmall.rows,
                                         maskSmall.cols, maskSmall.rows)));

        if (saveVideo) writer.write(canvas);

        cv::imshow("lightbar", canvas);
        char key = (char)cv::waitKey(30);    // imshow 后必须 waitKey 才刷新
        if (key == 27) break;                // ESC

        ++index;
    }

    std::cout << "processed " << index << " / " << total << " frames";
    if (missFrames) std::cout << "   (frames without full quad: " << missFrames << ")";
    std::cout << "\n";
    std::cout << "pose solved on " << poseFrames << " frames\n";
    if (saveVideo) std::cout << "result video: " << outPath << "\n";

    cap.release();
    if (saveVideo) writer.release();         // release 才会写完文件尾
    cv::destroyAllWindows();
    return 0;
}
