#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// 第 7 课候选方案：掩码(宽松) -> 候选 -> 按"成对"逻辑选出真正的一对
// 配对条件：
//   1) 两个候选中心 y 差 < max(8, 0.15*平均高度)
//   2) 尺寸比（面积）在 0.5 ~ 2.0 之间
//   3) 倾斜角差 < 25 度（换算到 -90..90）
//   4) 水平间距 > 两灯条宽度的 0.5 倍（保证确实是左右并排）
// 先按面积排序，以最大者为种子，找它最匹配的搭档。

struct Bar {
    cv::RotatedRect rect;
    double area = 0;
    float longSide = 0, shortSide = 0;
    double aspect = 0;
    cv::Point2f top, bottom;
};

static cv::Mat makeMask(const cv::Mat& f) {
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

static void endpointOf(const cv::RotatedRect& rr, cv::Point2f& top, cv::Point2f& bottom) {
    cv::Point2f p[4];
    rr.points(p);
    std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    top    = cv::Point2f((p[0].x + p[1].x) * 0.5f, (p[0].y + p[1].y) * 0.5f);
    bottom = cv::Point2f((p[2].x + p[3].x) * 0.5f, (p[2].y + p[3].y) * 0.5f);
}

// 角度归一化到 -90..90
static double normAngle(float angle) {
    double a = angle;
    while (a <= -90.0) a += 180.0;
    while (a >   90.0) a -= 180.0;
    return a;
}

static std::vector<Bar> pickPair(std::vector<Bar>& cand) {
    if (cand.size() < 2) return cand;
    std::sort(cand.begin(), cand.end(), [](const Bar& a, const Bar& b) { return a.area > b.area; });

    const Bar seed = cand[0];
    int best = -1;
    double bestScore = 1e18;

    for (size_t i = 1; i < cand.size(); ++i) {
        double dy = std::fabs(cand[i].rect.center.y - seed.rect.center.y);
        double areaRatio = cand[i].area / seed.area;
        double dAngle = std::fabs(normAngle(cand[i].rect.angle) - normAngle(seed.rect.angle));
        double dx = std::fabs(cand[i].rect.center.x - seed.rect.center.x);
        double minDx = 0.5 * (seed.shortSide + cand[i].shortSide);

        bool okY = dy < std::max(8.0, 0.15 * (seed.rect.size.height + cand[i].rect.size.height) * 0.5);
        bool okA = areaRatio > 0.5 && areaRatio < 2.0;
        bool okAng = dAngle < 25.0;
        bool okDx = dx > minDx;

        double score = dy * 2.0 + dAngle + std::fabs(1.0 - areaRatio) * 10.0;
        if (okY && okA && okAng && okDx && score < bestScore) {
            bestScore = score;
            best = (int)i;
        }
    }

    std::vector<Bar> out;
    out.push_back(seed);
    if (best >= 0) out.push_back(cand[best]);
    // 左边在前
    if (out.size() == 2 && out[1].rect.center.x < out[0].rect.center.x) std::swap(out[0], out[1]);
    return out;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "../video/装甲板视频.avi";
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) { std::cerr << "cannot open\n"; return 1; }

    cv::Mat frame;
    int idx = 0, n0 = 0, n1 = 0, n2 = 0;
    std::vector<int> bad;
    double sumLong = 0; int cnt = 0;

    while (cap.read(frame)) {
        cv::Mat mask = makeMask(frame);
        std::vector<std::vector<cv::Point>> cs;
        std::vector<cv::Vec4i> h;
        cv::findContours(mask.clone(), cs, h, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<Bar> cand;
        for (size_t i = 0; i < cs.size(); ++i) {
            Bar b;
            b.area = cv::contourArea(cs[i]);
            if (b.area < 10) continue;
            b.rect = cv::minAreaRect(cs[i]);
            b.longSide  = std::max(b.rect.size.width, b.rect.size.height);
            b.shortSide = std::min(b.rect.size.width, b.rect.size.height);
            if (b.shortSide < 1e-6f) continue;
            b.aspect = b.longSide / b.shortSide;
            if (b.aspect < 1.5) continue;
            if (b.longSide < 12) continue;
            cand.push_back(b);
        }

        std::vector<Bar> pair = pickPair(cand);
        if (pair.size() == 0) { ++n0; bad.push_back(idx); }
        else if (pair.size() == 1) { ++n1; bad.push_back(idx); }
        else { ++n2; sumLong += pair[0].longSide + pair[1].longSide; ++cnt; }
        ++idx;
    }

    printf("frames=%d  pair==0:%d  pair==1:%d  pair==2:%d\n", idx, n0, n1, n2);
    printf("avg bar long (paired) = %.1f\n", cnt ? sumLong / cnt : 0.0);
    printf("frames without a pair:");
    for (size_t i = 0; i < bad.size() && i < 30; ++i) printf(" %d", bad[i]);
    printf("%s\n", bad.size() > 30 ? " ..." : "");
    return 0;
}
