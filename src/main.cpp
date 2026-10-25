// ============================================================================
// 任务二：装甲板灯条识别 + 位姿解算（选做）
//
// 【整体流程】
//   必做四步：
//     ① 掩码 + 二值化 ：cv::threshold          -> 只留灯条像素
//     ② 识别灯条轮廓   ：cv::findContours       -> 提取候选
//     ③ 算上下端点     ：cv::minAreaRect + 几何 -> 每根灯条的上下端点
//     ④ 四端点连矩形   ：cv::line               -> 画出装甲板框
//   选做两步：
//     ⑤ 位姿解算       ：cv::solvePnP           -> 装甲板相对相机的 R、t
//     ⑥ 画三维坐标轴   ：cv::projectPoints      -> 把 FLU 坐标系画到图上
//
// 【位姿解算的核心概念（汇报要讲）】
//   * 位姿 = 位置 + 姿态 = 6 个自由度（3 个平移 + 3 个旋转）
//   * 单张图丢失深度：同一个物体在不同距离可以投出相同的像素位置。
//     所以必须借助"已知物理尺寸"才能反推距离 —— 这就是要定义 3D 点的原因。
//   * 投影方程： u = fx * (X/Z) + cx    （除以 Z 就是深度信息被消掉的地方）
//   * 旋转只有 3 个独立参数：3x3 矩阵有 9 个元素，但有 6 个约束
//     （3 列单位长度 + 3 对两两垂直）-> 9-6 = 3。
//     所以 solvePnP 返回 3 个数的旋转向量 rvec（轴角表示），而不是 9 个数的矩阵。
// ============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/opencv.hpp>

// 一根灯条：矩形 + 面积 + 尺寸 + 上下端点
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
// 选做 A：相机内参 K —— 相机自己的"属性表"
//
// 只需要记住 4 个数，其余 5 个（那些 0 和 1）是固定结构，不是测出来的：
//
//     K = [ fx   0   cx ]      fx = 928.13   焦距(x方向)，单位是【像素】
//         [ 0    fy  cy ]      fy = 930.14   焦距(y方向)，单位是【像素】
//         [ 0    0    1  ]     cx = 377.57   主点x —— 光轴打在图像上的 x 坐标
//                              cy = 283.89   主点y —— 光轴打在图像上的 y 坐标
//
// 【光轴是什么】穿过镜头中心、垂直于图像平面的那条直线，也就是相机"正视"的方向。
//   光轴与图像平面的交点就是主点 (cx, cy)。理论上应在图像正中心
//   (720/2, 540/2) = (360, 270)，实际偏了约 17 / 14 像素，是制造公差。
//
// 【为什么焦距是 928 这种数字，而不是"50毫米"】
//   这里的焦距【单位是像素】，不是毫米。
//   物理换算： f_像素 = f_毫米 / 每个像素的毫米数
//   例如像素尺寸 3 微米(0.003mm) -> f_毫米 = 928 * 0.003 ≈ 2.8 mm
//   也就是机器人上那种很小的镜头。
//
// 【投影公式（核心）】
//   u = fx * (X / Z) + cx
//   v = fy * (Y / Z) + cy
//   注意那个"除以 Z" —— 深度信息就是在这里被除掉的，
//   这正是"单目图像丢失深度"的数学来源。
//
// ★ OpenCV 相机坐标系：x 向右、y 向下、z 向前（指向场景）。
// ============================================================================
static cv::Mat cameraMatrix() {
    // cv::Mat_<double> 是 Mat 的模板版本，编译期就知道元素是 double，
    // 所以可以用 << 初始化。
    // ★ << 在这里【不是左移】，是 OpenCV 重载的"依次填数"，而且是【按行填】。
    // ★ 整个表达式要包一层括号，否则不能直接 return。
    return (cv::Mat_<double>(3, 3) <<
        9.28130989e+02, 0.0,            3.77572945e+02,   // 第 1 行：fx, 0, cx
        0.0,            9.30138391e+02, 2.83892859e+02,   // 第 2 行：0, fy, cy
        0.0,            0.0,            1.0);             // 第 3 行：0, 0, 1
}

// ============================================================================
// 选做 A2：畸变系数 —— 真实镜头的误差修正
//
// 【为什么需要】针孔模型假设光线走直线，但真实镜头是玻璃，会折射，导致：
//   径向畸变：越靠图像边缘，像素被"掰"得越明显（因为边缘光线入射角大）
//   切向畸变：镜片和传感器不完全平行，图像被轻微剪切
//
// 【★ 顺序必须记牢，OpenCV 规定的顺序是】
//     索引:  k1    k2    p1    p2    k3
//     类型:  径向  径向  切向  切向  径向
//   注意 p1/p2（切向）【夹在】k2 和 k3 中间！
//   填错顺序【不报错、不崩溃】，只是位姿算得不准 —— 这是最讨厌的一类错误。
//
// 【径向畸变公式】
//   r_畸变后 = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
//   k1 是主项，k2 是二阶修正，k3 只影响最边缘（因为它乘 r^6）
//
// 【这组数对本文相机意味着什么】
//   k1 = -0.254（负值 -> 边缘被往里压缩，典型桶形畸变）
//   p1=0.0037, p2=-0.0011（比径向小两个数量级 -> 镜片装得很正，切向可忽略）
//   实测图像角点处半径压缩约 4%（450 像素的角点偏约 18 像素），中心几乎无畸变。
//
// 【重要实践结论】做 PnP 时【不需要】自己先 undistort 一遍，
//   solvePnP 和 projectPoints 都接受 distCoeffs 参数，内部会处理。
// ============================================================================
static cv::Mat distortionCoeffs() {
    return (cv::Mat_<double>(1, 5) <<
        -2.54433647e-01,   // k1  径向（主项）
         5.69431382e-01,   // k2  径向
         3.65405229e-03,   // p1  切向  ← 注意切向夹在中间
        -1.09433818e-03,   // p2  切向
        -1.33846840e+00);  // k3  径向（只影响最边缘）
}

// ============================================================================
// 选做 B：装甲板 4 个灯条端点的 3D 坐标（单位：毫米）
//
// 【★ 这是唯一需要我们自己拍板的数据】
//   PnP 的输入是"3D 点 <-> 2D 点"的对应关系。
//   2D 点我们能从图像里测（灯条端点的像素坐标），
//   但 3D 点必须由我们告诉程序："这 4 个点在现实中各相距多少毫米"。
//   这就是"用已知尺寸换深度"的那一步。
//
// 【坐标系：任务书要求的 FLU 轴序】
//   原点 = 装甲板中心
//   x 轴 = 指向装甲板前方（朝向相机）
//   y 轴 = 指向装甲板左侧
//   z 轴 = 指向装甲板上方
//
// 【为什么 4 个点的 x 全是 0】
//   装甲板是一个平面，4 个端点都在这个平面上，前后没有厚度差 -> x = 0。
//   所以这 4 个点只分布在 y-z 平面上，构成一个矩形。
//   注意：这一点也正是"平面 PnP 朝向歧义"的根源（见 estimatePose 里的说明）。
//
// 【⚠ 尺寸按队伍装甲板取】
//   横向 W（两灯条中心距）= 130 mm
//   纵向 L（灯条长度）    = 50 mm
//   影响分析：
//     距离的绝对值 —— 成正比。W 估小 10%，算出的距离也小 10%。
//     朝向（旋转）—— 几乎不影响。旋转靠 4 个点的相对位置关系确定，对绝对尺度不敏感。
//
// 【★★ 顺序必须和 2D 点严格一一对应，这是最容易错的地方】
//   [0] 左灯条上端点   [1] 右灯条上端点
//   [2] 右灯条下端点   [3] 左灯条下端点
//   如果顺序搞错，PnP【不报错】，但解出的位姿是扭曲的，只能靠看图发现。
//   注：这个顺序和画四边形时的 {L.top, R.top, R.bottom, L.bottom} 完全一致，
//       同一组 4 个点既画四边形又喂 PnP，用同一套顺序可以少一个出错机会。
//
// 【正负号怎么记】
//   y 指向【左】-> 左边是 +y，右边是 -y
//   z 指向【上】-> 上面是 +z，下面是 -z
// ============================================================================
static std::vector<cv::Point3f> armorObjectPoints() {
    const float L2 = 50.0f  / 2.0f;   // 半灯条长 = 25 (纵向 z)
    const float W2 = 130.0f / 2.0f;   // 半间距   = 65 (横向 y)

    return {
        { 0.0f,  W2,  L2 },   // [0] 左灯条 上端点   y=+65(左)  z=+25(上)
        { 0.0f, -W2,  L2 },   // [1] 右灯条 上端点   y=-65(右)  z=+25(上)
        { 0.0f, -W2, -L2 },   // [2] 右灯条 下端点   y=-65(右)  z=-25(下)
        { 0.0f,  W2, -L2 },   // [3] 左灯条 下端点   y=+65(左)  z=-25(下)
    };
}

// ============================================================================
// 第 ① 步：掩码 + 二值化
//
// 【为什么不能只用一个灰度阈值】
//   装甲板中间白色数字的灰度和灯条一样高，threshold 会把它一起选进来。
//   实测：灯条像素 B=40 G=62 R=255（R-G=193）；白字 B=100 G=96 R=96（R-G=0）。
//   所以必须用【颜色】区分：灯条"红得厉害"，白字"白得均匀"。
//
// 【阈值的关键：落在"两侧都没东西"的空档里】
//   灯条     R-G = 100~190   gray = 60~172    两个条件都过 -> 保留
//   地面反光 R-G = 40~80     gray = 20~45     过了颜色关，但【过不了亮度关】-> 滤掉
//   白色数字 R-G ≈ 0         gray ≈ 100       过不了颜色关 -> 滤掉
//   背景     R-G ≈ 0         gray < 20        都不过 -> 滤掉
//
// 【★ 亮度门槛是这个场景最强的判别特征】
//   灯条是【自己发光】的，地面反光只是【反射】，能量差一个量级。
//   把亮度提到 gray>50 之后，反光自然消失，101 帧每帧恰好只剩 2 个候选。
// ============================================================================
static cv::Mat makeMask(const cv::Mat& frame) {
    std::vector<cv::Mat> ch;
    // ★ OpenCV 的通道顺序是 BGR 不是 RGB：ch[0]=B  ch[1]=G  ch[2]=R
    cv::split(frame, ch);

    // ---- 条件一：颜色（红）----
    cv::Mat diff, maskColor;
    cv::subtract(ch[2], ch[1], diff);          // R - G
    // threshold 五个参数：(源, 目标, 阈值, 最大值, 规则)
    // THRESH_BINARY 规则： > 阈值 的变 255（白），其余变 0（黑）
    cv::threshold(diff, maskColor, 20, 255, cv::THRESH_BINARY);

    // ---- 条件二：亮度 ----
    cv::Mat gray, maskBright;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);   // 三通道 -> 单通道灰度
    cv::threshold(gray, maskBright, 50, 255, cv::THRESH_BINARY);

    // ---- 两个条件必须【同时】满足 ----
    // bitwise_and 名字叫"按位与"，但在只有 0/255 的二值图上等价于"逻辑与"：
    //   255 & 255 = 255 (要)    255 & 0 = 0 (丢)    0 & 255 = 0 (丢)
    cv::Mat mask;
    cv::bitwise_and(maskColor, maskBright, mask);

    // ---- 形态学开运算：去掉孤立的噪点小块 ----
    // 开运算 = 先腐蚀后膨胀：所有白块先缩一圈，小碎块在腐蚀时就彻底消失，膨胀也长不回来。
    // ★ 用 2x2 而不是 3x3：远处灯条只有 2~3 像素宽，3x3 的腐蚀会把它整体吃掉。
    // 实测：第 0 帧用 3x3 时 55 像素 -> 0 像素（灯条全没了）。
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// ============================================================================
// 第 ③ 步：从旋转矩形算灯条的上下端点
//
// 【核心概念：端点不是角点】
//   矩形有 4 个角点，但灯条是一个"柱状发光体"，
//   我们要的是它【长轴两端的中心】，也就是上下两条短边的中点。
//
//       ●─────────●      4 个 ● 是角点
//       ○─────────○      2 个 ○ 才是端点（短边中点）
//
//   为什么必须是中点：任务书要 4 个端点（2 根灯条 x 2），
//   所以每根灯条只能贡献 2 个点。
//
// 【★ 为什么必须先排序】
//   RotatedRect::points() 返回的 4 个点【顺序不保证】，取决于矩形的角度。
//   不排序直接用 pts[0]、pts[2]，端点会在帧之间乱跳。
//
// 【⚠ 这个做法的隐含假设】
//   灯条接近竖直。4 个角点的 y 值天然分成"上两个/下两个"两簇，
//   排序后前两个是上边、后两个是下边。
//   如果灯条横躺，y 值会交错，这个办法就失效了（当前场景成立）。
// ============================================================================
static void getEndpoints(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);                        // 往 p 里填入 4 个角点

    // std::sort(起点, 终点, 比较规则)
    //   终点是 p+4：C++ 迭代器区间是【左闭右开】[first,last)，
    //   p+4 指向"最后一个元素的下一个位置"。写 p+3 会漏掉一个且不报错。
    // 比较器用 lambda：升序（y 小的在前 = 图像上方）
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });

    // 中点公式：两点各分量相加除以 2
    // 写 *0.5f 而不是 /2：乘法更快，末尾 f 表示 float 常量，避免多余类型转换
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

// ============================================================================
// 第 ②③ 步：找轮廓 -> 筛候选 -> 取面积最大的两根 -> 算端点
// ============================================================================
static std::vector<LightBar> findLightBars(const cv::Mat& mask) {
    // 嵌套 vector 的读法：从里往外
    //   里层 vector<Point>     = 一个轮廓（一串点）
    //   外层 vector<...>       = 一堆轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;    // Vec4i = 4 个 int，存轮廓父子关系（本处用不到但接口要求）

    // ★ 传 mask.clone()：findContours 会修改输入图
    // RETR_EXTERNAL      = 只要最外层轮廓（灯条是实心白块，内部没洞）
    // CHAIN_APPROX_SIMPLE = 压缩共线点，一条直线只留端点，省内存
    cv::findContours(mask.clone(), contours, hierarchy,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> cand;
    for (size_t i = 0; i < contours.size(); ++i) {
        LightBar b;
        b.area = cv::contourArea(contours[i]);
        if (b.area < 10) continue;           // 太小 = 噪点

        b.rect = cv::minAreaRect(contours[i]);   // 拟合可旋转的最小外接矩形

        // ★ minAreaRect 的 size.width 不保证是短边，取决于内部怎么排！
        //   所以判长宽比【必须】用 max/min，否则可能算出小于 1 的值。
        b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
        b.shortSide = std::min(b.rect.size.width, b.rect.size.height);

        // 防除零：浮点数不用 ==0 判断（有精度误差），用 < 1e-6
        if (b.shortSide < 1e-6f) continue;

        double aspect = b.longSide / b.shortSide;
        if (aspect < 1.5) continue;          // 不够细长（圆形光斑会被滤掉）
        if (b.longSide < 12) continue;       // 太短
        cand.push_back(b);
    }

    // 按面积【降序】排序：> 表示大的排前面
    // 比较器参数用 const LightBar&，避免每次比较都复制整个结构体
    std::sort(cand.begin(), cand.end(),
              [](const LightBar& a, const LightBar& b) { return a.area > b.area; });

    // 自检：候选超过 2 个时，检查前二是否真的"像一对灯条"
    //   同一水平线（dy 小）+ 尺寸相当（面积比接近1）+ 大致平行（角度差小）
    // 不像就只保留最大的一个 —— 宁可少画（不画四边形），也不要画错位置。
    if (cand.size() >= 3) {
        const LightBar& a = cand[0];
        const LightBar& b = cand[1];
        double dy = std::fabs(a.rect.center.y - b.rect.center.y);
        double hAvg = 0.5 * (a.rect.size.height + b.rect.size.height);
        double areaRatio = b.area / a.area;
        double dAngle = std::fabs(normAngle(a.rect.angle) - normAngle(b.rect.angle));
        // std::max(12, 0.25*hAvg)：远处灯条很小时 0.25*hAvg 只有 2 像素太严，
        // 用绝对值兜底，保证至少容许 12 像素
        bool plausible = (dy < std::max(12.0, 0.25 * hAvg))
                      && (areaRatio > 0.35 && areaRatio < 2.8)
                      && (dAngle < 35.0);
        if (!plausible) cand.resize(1);      // resize 缩小 = 截断，只留第 0 个
    }

    if (cand.size() > 2) cand.resize(2);     // 只留前两名 = 两根灯条

    // 对留下来的算端点（筛选时用不到端点，所以推迟到这里算）
    for (size_t i = 0; i < cand.size(); ++i) {
        getEndpoints(cand[i].rect, cand[i].top, cand[i].bottom);
    }
    return cand;
}

// ============================================================================
// 选做 C：位姿解算 —— cv::solvePnP
//
// 【solvePnP 在做什么】
//   输入：若干组「3D 点 <-> 2D 点」的对应关系 + 相机内参 + 畸变
//   输出：把 3D 点变到 2D 的那个变换，即 6 个自由度的位姿
//
// 【6 个自由度怎么来的】
//   平移 tvec：3 个数（x/y/z 各偏多少毫米）
//   旋转 rvec：3 个数（轴角表示：方向=旋转轴，长度=旋转角度弧度）
//   -> 合计 6
//
// 【为什么至少要 4 个点（3 个也能解，但 4 个更稳）】
//   未知数 6 个，每个点提供 2 个方程（u 一个、v 一个）
//   3 个点 = 6 方程，刚好能解；4 个点 = 8 方程，超定，更稳。
//   我们正好有 4 个端点 —— 任务书先让算端点的巧思就在这里。
//
// 【★ 为什么用 SOLVEPNP_IPPE 而不是默认的 ITERATIVE】
//   我们的 4 个点 x 坐标全是 0，都在装甲板这个平面上，是【共面点】。
//   IPPE（Infinitesimal Plane-based Pose Estimation）专为共面目标设计，4 点即可。
//   默认的 SOLVEPNP_ITERATIVE 对非共面要求 6 个点，共面情况下可能不稳。
//
// 【useExtrinsicGuess = false】
//   表示"不用传进来的 rvec/tvec 当初值迭代"。
//   如果设 true，就可以拿上一帧的结果做初值来加快收敛、抑制跳变（可优化的点）。
// ============================================================================
static bool estimatePose(const std::vector<LightBar>& bars,
                         cv::Vec3d& rvec, cv::Vec3d& tvec) {
    if (bars.size() != 2) return false;      // 不够两根灯条，解不了

    // ★★ 2D 点的顺序必须和 armorObjectPoints() 里 3D 点的顺序【严格一一对应】
    //   3D[0] (0,+W2,+L2) <-> 2D[0] 左灯条上端点
    //   3D[1] (0,-W2,+L2) <-> 2D[1] 右灯条上端点
    //   3D[2] (0,-W2,-L2) <-> 2D[2] 右灯条下端点
    //   3D[3] (0,+W2,-L2) <-> 2D[3] 左灯条下端点
    //   顺序错了不报错，但位姿会扭曲。
    //   bars[0] 是左灯条、bars[1] 是右灯条（main 里已按 x 排好）。
    std::vector<cv::Point2f> imgPts = {
        bars[0].top, bars[1].top, bars[1].bottom, bars[0].bottom
    };

    // solvePnP 八个参数，前六个必填：
    //   objectPoints 3D点 / imagePoints 2D点 / cameraMatrix 内参 / distCoeffs 畸变
    //   rvec 输出旋转向量 / tvec 输出平移向量
    bool ok = cv::solvePnP(armorObjectPoints(), imgPts,
                           cameraMatrix(), distortionCoeffs(),
                           rvec, tvec,
                           false, cv::SOLVEPNP_IPPE);

    // ★ 必须检查返回值：失败时 rvec/tvec 是未定义的垃圾值
    if (!ok) return false;

    // ========================================================================
    // ★★★ 平面目标的朝向歧义矫正（这是本项目最有价值的一个坑）
    //
    // 【现象】
    //   solvePnP 返回 true，解出的距离也完全合理（1.74 m），
    //   但把三维轴画出来一看：红色 x 轴指向"远离相机"，
    //   等于认为我们看到的是装甲板的【背面】。
    //
    // 【根因】
    //   我们给的 4 个 3D 点 x 坐标【全是 0】（都在装甲板平面上）。
    //   把这 4 个点的 x 全取负，点集完全不变！
    //   所以 PnP 在几何上【无法判断这个平面的法向朝哪边】——
    //   正着放和反着放，4 个点坐标一摸一样，投影也一摸一样。
    //   这是平面目标 PnP 的【固有二义性】，不是 OpenCV 的 bug，
    //   和"内克尔立方体"那个视觉错觉是同一个数学问题。
    //   实测两个 IPPE 解的 x 轴 z 分量都是 +0.95 / +0.97（都指向远离相机）。
    //
    // 【怎么解 —— 用先验知识】
    //   物理事实：比赛里装甲板正面一定朝向相机。
    //   判据：物体 x 轴在相机坐标系的 z 分量 > 0 -> 指向远离相机 -> 解反了。
    //
    // 【为什么需要 Rodrigues 来回转】
    //   rvec 是 3 个数的轴角表示，【不能直接做矩阵乘法】；
    //   要取"某一根轴的方向"必须用旋转矩阵 R，而 R 的【列就是各轴的方向】。
    //   改完 R 之后必须转回 rvec，因为 projectPoints 只接受 rvec。
    //   ⚠ 忘了转回去 = 改了等于没改，而且完全不报错。
    // ========================================================================
    cv::Mat R;
    cv::Rodrigues(rvec, R);                  // 旋转向量 -> 旋转矩阵

    // R 的列 = 物体各轴在相机系中的方向：
    //   列0 = 物体 x 轴    列1 = 物体 y 轴    列2 = 物体 z 轴
    // R.at<double>(行, 列) 是"先行后列"。
    // 所以 R.at<double>(2,0) = 物体 x 轴方向的【z 分量】。
    //   > 0 : 指向相机后方（远离我们）-> 判定成"背面朝向相机"-> 解反了
    //   < 0 : 指向相机前方（朝向我们）-> 正确
    if (R.at<double>(2, 0) > 0.0) {
        // 翻转方式：右乘 diag(-1,-1,1)，等价于【绕物体 z 轴转 180°】
        //   -> x 反号（转成朝向相机）
        //   -> y 反号
        //   -> z 不变（保证"上"还是上）
        // 为什么必须 x、y 同时反：det(diag(-1,-1,1)) = (-1)*(-1)*1 = +1，
        //   行列式为 +1 才是【纯旋转】。只反一个（det=-1）就是镜像，
        //   会把右手坐标系变成左手系，画出来的轴会乱。
        //
        // 副作用（是对的，不是 bug）：翻转后 y 轴从指向图像左侧变成指向右侧。
        //   因为正对着看装甲板时，它的"左手"在观察者的右手边。
        //   任务书参考图里绿色 y 轴也是朝右的，和我们的结果一致。
        cv::Mat flip = (cv::Mat_<double>(3, 3) <<
            -1.0,  0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0,  0.0, 1.0);

        // R * flip 是矩阵乘法（右乘），效果是"把 R 的每一列乘上对应的对角元"：
        //   第0列 x (-1)、第1列 x (-1)、第2列 x (+1)
        R = R * flip;

        // ★ 必须转回旋转向量，否则后面 projectPoints 用的还是旧的 rvec
        cv::Rodrigues(R, rvec);              // 旋转矩阵 -> 旋转向量
    }
    return true;
}

// ============================================================================
// 选做 D：画 FLU 三维坐标轴
//
// 【核心思路：projectPoints 是 solvePnP 的逆运算】
//   solvePnP      : 2D点 + 3D点        -> 位姿
//   projectPoints : 位姿 + 3D点        -> 2D点
//
// 所以画轴分三步：
//   1) 在【物体坐标系】里定义 4 个 3D 点：1 个原点 + 3 个轴端点
//   2) 用刚解出的位姿把它们投影回图像
//   3) 在图像上从原点向三个端点画箭头
//
// 【★ 为什么原点也必须投影】
//   任务书要求"坐标轴原点为装甲板中心点"，而中心在图像上的位置是变化的
//   （装甲板会移动）。不能假设原点在图像某个固定位置，必须让 projectPoints 算。
//
// 【为什么红轴看起来常常很短 —— 透视缩短】
//   x 轴指向装甲板前方，几乎正对着相机。
//   一个"正对着你"的箭头，投影出来只剩极短一段甚至一个点。
//   z（上）和 y（侧）垂直于视线，所以能完整投影。
//   所以：红轴的长短 = 装甲板相对相机歪了多少。这是对的现象。
// ============================================================================
static void drawAxes(cv::Mat& canvas, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                     float axisLen, int thickness) {
    // 物体坐标系下的 4 个点：原点 + 三个轴端点
    std::vector<cv::Point3f> axis3d = {
        { 0.0f,    0.0f,    0.0f },        // [0] 原点 = 装甲板中心
        { axisLen, 0.0f,    0.0f },        // [1] x 轴端点（前方）
        { 0.0f,    axisLen, 0.0f },        // [2] y 轴端点（左侧）
        { 0.0f,    0.0f,    axisLen },     // [3] z 轴端点（上方）
    };

    // 把 3D 点按位姿投影回图像 -> 得到 4 个像素坐标
    // 注意这里 rvec/tvec 是【输入】（已知），和 solvePnP 里正好相反
    std::vector<cv::Point2f> axis2d;
    cv::projectPoints(axis3d, rvec, tvec, cameraMatrix(), distortionCoeffs(), axis2d);

    // BGR 颜色：x 红、y 绿、z 蓝（和 Blender / ROS RViz 等三维软件的约定一致）
    // 注意 Scalar 是 BGR 顺序：Scalar(0,0,255) 才是红色
    const cv::Scalar color[3] = {
        cv::Scalar(0,   0, 255),      // x 红
        cv::Scalar(0, 255,   0),      // y 绿
        cv::Scalar(255, 0,   0),      // z 蓝
    };
    const char* name[3] = { "x", "y", "z" };

    // 把旋转向量转成矩阵，后面要把轴端点转到相机坐标系算坐标标签
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // 拆成普通 C 数组：R.at<double>() 每次都要类型检查，9 次调用不划算；
    // 而且 r[a][b]*P.x + ... 一眼能看出是矩阵乘法的展开式。
    double r[3][3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            r[a][b] = R.at<double>(a, b);

    // ---- 第一轮：先画全部三条箭头 ----
    // ★ 为什么分两轮：画图是"后画覆盖先画"。
    //   如果在一个循环里"画箭头->画标签"逐个做，后面轴的箭头会盖住前面轴的标签。
    cv::Point o = axis2d[0];                 // 原点的像素位置，三条轴的公共起点
    for (int i = 0; i < 3; ++i) {
        // arrowedLine(图, 起点, 终点, 颜色, 线宽, 线型, shift, 箭头长度占比)
        //   cv::LINE_AA = 抗锯齿，线条平滑
        //   shift = 0   坐标小数位数
        //   tipLength = 0.25  箭头占整条线的 25%（0.2~0.3 比较好看）
        cv::arrowedLine(canvas, o, axis2d[i + 1], color[i], thickness,
                        cv::LINE_AA, 0, 0.25);
    }

    // ---- 第二轮：再画全部标签（保证标签在最上层）----
    for (int i = 0; i < 3; ++i) {
        cv::Point tip = axis2d[i + 1];       // 第 i 条轴的端点（i+1 因为 [0] 是原点）

        // 该轴端点在【相机坐标系】下的位置：P_cam = R * P_obj + t，换算成米
        // 这是手动展开矩阵乘法（零堆分配）：
        //   cx = r00*Px + r01*Py + r02*Pz + tx
        const cv::Point3f& P = axis3d[i + 1];
        double cx = r[0][0]*P.x + r[0][1]*P.y + r[0][2]*P.z + tvec[0];
        double cy = r[1][0]*P.x + r[1][1]*P.y + r[1][2]*P.z + tvec[1];
        double cz = r[2][0]*P.x + r[2][1]*P.y + r[2][2]*P.z + tvec[2];

        // snprintf 是带长度限制的安全版 sprintf：
        //   sizeof label 告诉它缓冲区多大，防止写越界
        //   %.1f 保留 1 位小数；/1000.0 把毫米换成米（1.7 比 1738.7 好读）
        char label[64];
        std::snprintf(label, sizeof label, "%s(%.1f,%.1f,%.1f)",
                      name[i], cx / 1000.0, cy / 1000.0, cz / 1000.0);

        // 标签沿"原点->轴尖"方向再往外推 16 像素，避免三条轴的标签互相压住
        cv::Point d = tip - o;
        double dlen = std::sqrt((double)d.x * d.x + (double)d.y * d.y);
        cv::Point off = (dlen > 1.0)
                      ? cv::Point((int)(d.x / dlen * 16), (int)(d.y / dlen * 16))
                      : cv::Point(8, -8);

        // ★ 用 getTextSize 量出文字包围盒，把标签限制在画面内（否则会被边界切掉）
        //   参数顺序和 putText 前四个一致，最后一个 &baseline 输出"基线以下的高度"
        const double fscale = 0.55;
        const int    fthick = 2;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, 0, fscale, fthick, &baseline);

        // ★ putText 的坐标是文字的【左下角】（不是左上角）
        //   所以：文字顶部 = y - height，文字右边界 = x + width
        //   注意 canvas.cols 是宽、canvas.rows 是高
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
    // 三元运算符：有参数就用参数，没有就用默认值
    // 这里 && 的短路特性有安全意义：argc 不足时 argv[2] 不被求值，避免越界
    bool saveVideo = !(argc > 2 && std::string(argv[2]) == "--no-save");

    cv::VideoCapture cap(videoPath);
    // ★ 打开失败不报错、不抛异常，只是内部状态无效，必须自己检查
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << videoPath << "\n";
        return -1;
    }

    // ★ get() 永远返回 double，要按需强转
    //   fps 保持 double：像 29.97 这种帧率转 int 会截断，导致输出视频速度偏快
    const int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double fps    = (double)cap.get(cv::CAP_PROP_FPS);
    const int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    // 有些视频的 FPS 返回 0，用 0 去建 VideoWriter 会失败，所以给个兜底值
    const double useFps = (fps > 0.0 ? fps : 25.0);

    std::cout << "video : " << videoPath << "\n";
    std::cout << "size  : " << width << " x " << height
              << "   fps: " << fps << "   frames: " << total << "\n";

    // ---- 输出结果视频 ----
    // fourcc = Four Character Code，用四个字符表示视频编码格式。
    // MJPG + .avi 在 Linux 上最稳；mp4 需要 H.264 编码器，自编译的 OpenCV 常常没有。
    // 帧率、尺寸必须和原视频一致，否则输出的视频速度不对或花屏。
    cv::VideoWriter writer;
    std::string outPath = "../output/03_最终结果/result.avi";
    if (saveVideo) {
        writer.open(outPath, cv::VideoWriter::fourcc('M','J','P','G'), useFps,
                    cv::Size(width, height));
        // ★ VideoWriter 打开失败同样不报错，必须检查
        //   失败时把 saveVideo 设成 false —— 一个可选功能失败不该让整个程序挂掉
        if (!writer.isOpened()) {
            std::cerr << "warning: cannot open " << outPath << ", will not save\n";
            saveVideo = false;
        }
    }

    double scale = width / 640.0;    // 字号按画面宽度缩放（640 是基准宽度）
                                     // 注意 640.0 不是 640：整数除法会截断

    // ---- 显示窗口：放大到 1.8 倍 ----
    // ★ imshow 默认创建的是 WINDOW_AUTOSIZE 窗口，大小锁死成图片尺寸，拖不动。
    //   必须先 namedWindow 指定 WINDOW_NORMAL 才能自由缩放。
    // ★ 而且 namedWindow 必须在第一次 imshow 之前调用。
    // ★ resizeWindow 改的是【窗口】尺寸，图片数据不变 —— 输出视频仍是原尺寸。
    cv::namedWindow("lightbar", cv::WINDOW_NORMAL);
    double dispScale = 1.8;
    cv::resizeWindow("lightbar",
                     (int)(width  * dispScale),
                     (int)(height * dispScale));

    // 坐标轴长度（毫米）。只为好看，【不影响位姿解算结果】。
    // 取 200 比装甲板（130mm 宽）长一些，三条轴和标签才不会挤在一起。
    const float AXIS_LEN_MM = 200.0f;

    cv::Mat frame;
    int index = 0;
    int missFrames = 0;     // 统计没画出四边形的帧数
    int poseFrames = 0;     // 统计位姿解算成功的帧数

    // ★ while (cap.read(frame)) 一行完成"读"和"判断"：
    //   read() 既取下一帧填进 frame，又返回"读到了吗"。读完返回 false，循环结束。
    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<LightBar> bars = findLightBars(mask);

        // 按 x 分左右，保证连线是 左->右->右->左，不会交叉成蝴蝶结
        // ★ && 短路保证 bars 只有 1 个元素时不会去访问 bars[1]（越界不报错但读到垃圾）
        if (bars.size() == 2 && bars[1].rect.center.x < bars[0].rect.center.x)
            std::swap(bars[0], bars[1]);

        // ★ 必须 clone：cv::Mat 的赋值是【浅拷贝】（引用计数），
        //   canvas = frame 会让两者共享同一块图像数据，
        //   往 canvas 上画线会把原图也改了。
        cv::Mat canvas = frame.clone();

        // ---- 每根灯条：画自己的旋转矩形 + 上下端点 ----
        for (size_t i = 0; i < bars.size(); ++i) {
            cv::Point2f p[4];
            bars[i].rect.points(p);
            // (k+1)%4 让下标在 0~3 循环，最后一条边从 p[3] 回到 p[0]，闭合成矩形
            for (int k = 0; k < 4; ++k)
                cv::line(canvas, p[k], p[(k + 1) % 4], cv::Scalar(180, 180, 0), 1);
            // 线宽 -1 = 实心圆
            cv::circle(canvas, bars[i].top,    3, cv::Scalar(0, 255, 255), -1);  // 上端点 黄
            cv::circle(canvas, bars[i].bottom, 3, cv::Scalar(255, 0, 255), -1);  // 下端点 紫
        }

        // ---- 第 ④ 步：四端点按顺序连线画矩形 ----
        if (bars.size() == 2) {
            const LightBar& L = bars[0];    // const 引用 = 别名，零复制
            const LightBar& R = bars[1];
            // 顺序：左上 -> 右上 -> 右下 -> 左下（天然不自交）
            cv::Point2f quad[4] = { L.top, R.top, R.bottom, L.bottom };
            for (int i = 0; i < 4; ++i)
                cv::line(canvas, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 255), 2);
        } else {
            ++missFrames;
        }

        // ---- 第 ⑤⑥ 步（选做）：位姿解算 + 画三维坐标轴 ----
        // cv::Vec3d = 3 个 double 的向量，默认构造【不初始化】，但下面会检查返回值
        cv::Vec3d rvec, tvec;
        if (estimatePose(bars, rvec, tvec)) {    // 只有解算成功才画
            ++poseFrames;
            drawAxes(canvas, rvec, tvec, AXIS_LEN_MM, 2);

            // 每 20 帧打印一次（全打印会刷屏）
            // ★ tvec 的三个分量在相机坐标系里：x 右、y 下、z 前
            //   tvec[2] 就是相机到装甲板的距离（单位毫米）
            if (index % 20 == 0) {
                std::cout << "frame " << index
                          << "  t = (" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ") mm"
                          << "  dist = " << cv::norm(tvec) / 1000.0 << " m\n";
                          // cv::norm(vec) = 向量欧氏长度 √(x²+y²+z²)
            }
        }

        // 左上角显示帧号和检出数量
        // putText 的坐标是文字【左下角】
        cv::putText(canvas,
                    "frame " + std::to_string(index) + "   bars: " + std::to_string(bars.size()),
                    cv::Point(8, (int)(26 * scale)), 0, 0.7 * scale,
                    cv::Scalar(0, 255, 255), 2);

        // ---- 右下角嵌一个小掩码，方便看它干不干净 ----
        // ★ 为什么先转 3 通道：copyTo 要求源和目标通道数一致，
        //   canvas 是 3 通道彩色，mask 是 1 通道灰度，直接贴会崩。
        // ★ canvas(cv::Rect(...)) 是【感兴趣区域 ROI】，它和原图【共享内存】，
        //   所以往它上面 copyTo 就等于在画布的那块区域里贴图。
        //   Rect(x, y, 宽, 高)：x = width-w 贴右边界，y = height-h 贴下边界 -> 右下角
        cv::Mat maskBgr, maskSmall;
        cv::cvtColor(mask, maskBgr, cv::COLOR_GRAY2BGR);
        cv::resize(maskBgr, maskSmall, cv::Size(width / 4, height / 4));
        maskSmall.copyTo(canvas(cv::Rect(width - maskSmall.cols, height - maskSmall.rows,
                                         maskSmall.cols, maskSmall.rows)));

        // 写入结果视频（尺寸必须和 open 时声明的一致）
        if (saveVideo) writer.write(canvas);

        // ★ imshow 之后必须有 waitKey，否则窗口不刷新（一片灰）
        //   waitKey(30) 等 30 毫秒 ≈ 33fps；返回值是按键的 ASCII 码
        cv::imshow("lightbar", canvas);
        char key = (char)cv::waitKey(30);
        if (key == 27) break;      // 27 = ESC 的 ASCII 码

        ++index;
    }

    std::cout << "processed " << index << " / " << total << " frames";
    if (missFrames) std::cout << "   (frames without full quad: " << missFrames << ")";
    std::cout << "\n";
    std::cout << "pose solved on " << poseFrames << " frames\n";
    if (saveVideo) std::cout << "result video: " << outPath << "\n";

    cap.release();
    if (saveVideo) writer.release();   // release 会写完文件尾，不调用可能导致视频不完整
    cv::destroyAllWindows();
    return 0;
}
