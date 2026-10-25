#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

// ============================================================================
// 任务二：装甲板灯条识别 + 位姿解算（选做）
//
// 必做四步：
//   ① 掩码 + 二值化 ：cv::threshold          -> 只留灯条像素
//   ② 识别灯条轮廓   ：cv::findContours       -> 提取候选
//   ③ 算上下端点     ：cv::minAreaRect + 几何 -> 每根灯条的上下端点
//   ④ 四端点连矩形   ：cv::line               -> 画出装甲板框
//
// 选做两步：
//   ⑤ 位姿解算       ：cv::solvePnP           -> 装甲板相对相机的 R、t
//   ⑥ 画三维坐标轴   ：cv::projectPoints      -> 把 FLU 坐标系画到图上
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

// ============================================================================
// 选做 A：相机内参和畸变系数（任务书直接给出，原样抄，不要改）
//
// K 的 4 个有用数字：
//   fx = 928.13, fy = 930.14   -> 焦距（单位：像素）
//   cx = 377.57, cy = 283.89   -> 主点（光轴在图像上的位置）
// 其余 0 和 1 是固定结构。
//
// D 的 5 个数顺序固定：k1, k2, p1, p2, k3
//   k1/k2/k3 = 径向畸变（越靠边越明显），p1/p2 = 切向畸变
// ============================================================================
static cv::Mat cameraMatrix() {
    return (cv::Mat_<double>(3, 3) <<
        9.28130989e+02, 0.0,            3.77572945e+02,
        0.0,            9.30138391e+02, 2.83892859e+02,
        0.0,            0.0,            1.0);
}

static cv::Mat distortionCoeffs() {
    return (cv::Mat_<double>(1, 5) <<
        -2.54433647e-01,   // k1
         5.69431382e-01,   // k2
         3.65405229e-03,   // p1
        -1.09433818e-03,   // p2
        -1.33846840e+00);  // k3
}

// ============================================================================
// 选做 B：装甲板 4 个灯条端点的 3D 坐标（单位：毫米）
//
// 物体坐标系（PDF 要求的 FLU 轴序）：
//   原点 = 装甲板中心
//   x 轴 = 指向装甲板前方（也就是朝向相机）
//   y 轴 = 指向装甲板左侧
//   z 轴 = 指向装甲板上方
//
// 因为装甲板是一个平面，所以 4 个点的 x 都是 0，只在 y-z 平面上分布。
//
// 【尺寸是假设值，PDF 没给】按 RoboMaster 标准小装甲板取：
//   灯条长度 L = 55 mm   -> 上下端点相距 55
//   两灯条中心距 W = 135 mm -> 左右端点相距 135
// 这两个数只影响"解出来的距离的绝对尺度"，
// 对"朝向"（旋转）的影响很小 —— 画 3D 轴用足够了。
//
// ★ 顺序必须和 2D 点一致！下面注释里的顺序就是 main 里传 2D 点的顺序。
// ============================================================================
static std::vector<cv::Point3f> armorObjectPoints() {
    const float L2 = 55.0f  / 2.0f;   // 半灯条长
    const float W2 = 135.0f / 2.0f;   // 半间距

    return {
        { 0.0f,  W2,  L2 },   // [0] 左灯条 上端点
        { 0.0f, -W2,  L2 },   // [1] 右灯条 上端点
        { 0.0f, -W2, -L2 },   // [2] 右灯条 下端点
        { 0.0f,  W2, -L2 },   // [3] 左灯条 下端点
    };
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
    cv::findContours(mask.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        LightBar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 10) continue;

        b.rect = cv::minAreaRect(contours[i]);
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
        if (b.shortSide < 1e-6f) continue;

        double aspect = b.longSide / b.shortSide;
        if (aspect < 1.5) continue;
        if (b.longSide < 12) continue;
        cand.push_back(b);
    }

    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });

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

// ============================================================================
// 选做 C：位姿解算 + 画 FLU 三维坐标轴
//
// solvePnP：输入 4 组 "3D 点 <-> 2D 点" 对应关系，解出物体的位姿
//   objectPoints : 4 个 3D 点（毫米，物体坐标系）
//   imagePoints  : 4 个 2D 点（像素，图像坐标系）
//   cameraMatrix : 内参
//   distCoeffs   : 畸变
//   rvec / tvec  : 输出，旋转向量 + 平移向量
//
// projectPoints：把 3D 点按位姿投影回图像，用来画轴
// ============================================================================
static bool estimatePose(const std::vector<LightBar>& bars,
                         cv::Vec3d& rvec, cv::Vec3d& tvec) {
    if (bars.size() != 2) return false;

    // 2D 点的顺序必须和 armorObjectPoints() 里 3D 点的顺序严格对应
    //   [0] 左灯条 上端点  [1] 右灯条 上端点
    //   [2] 右灯条 下端点  [3] 左灯条 下端点
    std::vector<cv::Point2f> imgPts = {
        bars[0].top, bars[1].top, bars[1].bottom, bars[0].bottom
    };

    // ★ 4 个点都在装甲板平面上（共面），所以用 SOLVEPNP_IPPE（专为平面目标设计）
    bool ok = cv::solvePnP(armorObjectPoints(), imgPts,
                           cameraMatrix(), distortionCoeffs(),
                           rvec, tvec,
                           false, cv::SOLVEPNP_IPPE);
    if (!ok) return false;

    // ★ 平面目标的朝向歧义矫正
    //
    // 我们的 4 个 3D 点 x 坐标全是 0（都在装甲板这个平面上）。
    // 把所有点的 x 取负，3D 点集合完全不变 -> PnP 无法从几何上判断"正面朝哪边"。
    // 实测两个 IPPE 解都认为 x 轴朝"远离相机"，等于把装甲板当成了背面朝向相机。
    //
    // 用先验纠正：装甲板正面朝向相机（比赛里装甲板正对着我们）。
    // 判据：物体 x 轴在相机坐标系的 z 分量为正 = 指向远离相机 -> 需要翻转。
    // 翻转方式：绕物体 z 轴转 180°，即 x、y 同时反号、z 不变（仍是右手系）。
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    if (R.at<double>(2, 0) > 0.0) {
        cv::Mat flip = (cv::Mat_<double>(3, 3) <<
            -1.0,  0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0,  0.0, 1.0);
        R = R * flip;
        cv::Rodrigues(R, rvec);          // 旋转矩阵转回旋转向量
    }
    return true;
}

// 在图上画 FLU 三轴：x 红(前) / y 绿(左) / z 蓝(上)，原点 = 装甲板中心
static void drawAxes(cv::Mat& canvas, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                     float axisLen, int thickness) {
    std::vector<cv::Point3f> axis3d = {
        { 0.0f,      0.0f,      0.0f },        // [0] 原点（装甲板中心）
        { axisLen,   0.0f,      0.0f },        // [1] x 轴端点（前）
        { 0.0f,      axisLen,   0.0f },        // [2] y 轴端点（左）
        { 0.0f,      0.0f,      axisLen },     // [3] z 轴端点（上）
    };

    std::vector<cv::Point2f> axis2d;
    cv::projectPoints(axis3d, rvec, tvec, cameraMatrix(), distortionCoeffs(), axis2d);

    // BGR 颜色：x 红、y 绿、z 蓝（和常见三维软件的约定一致）
    const cv::Scalar color[3] = {
        cv::Scalar(0, 0, 255),      // x
        cv::Scalar(0, 255, 0),      // y
        cv::Scalar(255, 0, 0),      // z
    };
    const char* name[3] = { "x", "y", "z" };

    // 把旋转向量转成旋转矩阵（Rodrigues），后面要把 3D 点转到相机坐标系
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    double r[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            r[a][b] = R.at<double>(a, b);

    // 先画三条箭头（避免后画的箭头压住先画的标签）
    cv::Point o = axis2d[0];
    for (int i = 0; i < 3; ++i) {
        cv::arrowedLine(canvas, o, axis2d[i + 1], color[i], thickness, cv::LINE_AA, 0, 0.25);
    }

    // 再统一画标签，保证标签在最上层
    for (int i = 0; i < 3; ++i) {
        cv::Point tip = axis2d[i + 1];

        // 该轴端点在相机坐标系下的位置：P_cam = R * P_obj + t，换算成米
        const cv::Point3f& P = axis3d[i + 1];
        double cx = r[0][0]*P.x + r[0][1]*P.y + r[0][2]*P.z + tvec[0];
        double cy = r[1][0]*P.x + r[1][1]*P.y + r[1][2]*P.z + tvec[1];
        double cz = r[2][0]*P.x + r[2][1]*P.y + r[2][2]*P.z + tvec[2];

        char label[64];
        std::snprintf(label, sizeof label, "%s(%.1f,%.1f,%.1f)",
                      name[i], cx / 1000.0, cy / 1000.0, cz / 1000.0);

        // 标签沿"原点->轴尖"的方向再往外推一点，避免三条轴的标签互相压住
        cv::Point d = tip - o;
        double dlen = std::sqrt((double)d.x * d.x + (double)d.y * d.y);
        cv::Point off = (dlen > 1.0)
                      ? cv::Point((int)(d.x / dlen * 16), (int)(d.y / dlen * 16))
                      : cv::Point(8, -8);
        // 用 getTextSize 量出文字包围盒，把标签限制在画面内（否则会被右边界切掉）
        const double fscale = 0.55;
        const int    fthick = 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, 0, fscale, fthick, &baseline);

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
    const float AXIS_LEN_MM = 200.0f;     // 坐标轴长度（毫米），放大便于观察（只为好看，不影响解算）

    // ---- 显示窗口：用 WINDOW_NORMAL 才能自由缩放 ----
    // 视频本身保持原尺寸（720x540），这里只放大"显示窗口"，不影响输出视频
    cv::namedWindow("lightbar", cv::WINDOW_NORMAL);
    double dispScale = 1.8;                       // 窗口放大倍数
    cv::resizeWindow("lightbar",
                     (int)(width  * dispScale),
                     (int)(height * dispScale));

    cv::Mat frame;
    int index = 0;
    int missFrames = 0;
    int poseFrames = 0;

    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<LightBar> bars = findLightBars(mask);

        if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x)
            std::swap(bars[0], bars[1]);

        cv::Mat canvas = frame.clone();

        for (size_t i = 0; i < bars.size(); ++i) {
            cv::Point2f p[4];
            bars[i].rect.points(p);
            for (int k = 0; k < 4; ++k)
                cv::line(canvas, p[k], p[(k + 1) % 4], cv::Scalar(180, 180, 0), 1);
            cv::circle(canvas, bars[i].top,    3, cv::Scalar(0, 255, 255), -1);
            cv::circle(canvas, bars[i].bottom, 3, cv::Scalar(255, 0, 255), -1);
        }

        // ---- 第 ④ 步：四端点按顺序连线 ----
        if (bars.size() == 2) {
            const LightBar& L = bars[0];
            const LightBar& R = bars[1];
            cv::Point2f quad[4] = { L.top, R.top, R.bottom, L.bottom };
            for (int i = 0; i < 4; ++i)
                cv::line(canvas, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        } else {
            ++missFrames;
        }

        // ---- 第 ⑤⑥ 步（选做）：位姿解算 + 画三维坐标轴 ----
        cv::Vec3d rvec, tvec;
        if (estimatePose(bars, rvec, tvec)) {
            ++poseFrames;
            drawAxes(canvas, rvec, tvec, AXIS_LEN_MM, 2);

            // 打印距离和朝向（tvec 的 z 就是相机到装甲板的距离，单位 mm）
            if (index % 20 == 0) {
                std::cout << "frame " << index
                          << "  t = (" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ") mm"
                          << "  dist = " << cv::norm(tvec) / 1000.0 << " m\n";
            }
        }

        cv::putText(canvas,
                    "frame " + std::to_string(index) + "   bars: " + std::to_string(bars.size()),
                    cv::Point(8, (int)(26 * scale)), 0, 0.7 * scale,
                    cv::Scalar(0, 255, 255), 2);

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
    std::cout << "pose solved on " << poseFrames << " frames\n";
    if (saveVideo) std::cout << "result video: " << outPath << "\n";

    cap.release();
    if (saveVideo) writer.release();
    cv::destroyAllWindows();
    return 0;
}
