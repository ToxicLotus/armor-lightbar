#include <iostream>
#include <opencv2/opencv.hpp>

// 第 1 课：用数字看灯条
// 读一张图，打印几个关键位置的像素值，看清"灯条"和"白字"在数字上的区别。
                                                                          //argv=argument vector参数向量/参数数组（约定俗成）
int main(int argc, char** argv) {                                         //argc=argument count参数个数,char**argv=char* argv[]
    std::string path = (argc > 1) ? argv[1] : "images/frame_010.bmp";

    cv::Mat img = cv::imread(path);
    if (img.empty()) {
        std::cerr << "cannot read image: " << path << std::endl;
        return -1;
    }

    std::cout << "image: " << path << std::endl;
    std::cout << "size : " << img.cols << " x " << img.rows
              << "   (cols=width, rows=height)" << std::endl;
    std::cout << "channels: " << img.channels() << std::endl;

    // 像素访问是 img.at<...>(y, x)：y 是行(上下)，x 是列(左右)
    struct Probe { int y, x; const char* what; };                        //结构体定义一个新类型叫Probe
    Probe probes[] = {                                                   //在Probe类型里创建变量probes
        {250, 285, "left  light bar"},
        {250, 400, "right light bar"},
        {240, 310, "white digit on armor"},
        { 10,  10, "background"},
    };

    std::cout << "\n---- pixel values ----" << std::endl;
    for (const auto& p : probes) {                                       //for循环的新写法，遍历probes里的每一个元素
        cv::Vec3b v = img.at<cv::Vec3b>(p.y, p.x);                       // v[0]=B v[1]=G v[2]=R
        int B = v[0], G = v[1], R = v[2];
        std::cout << "  " << p.what << "  at(y=" << p.y << ", x=" << p.x << ")"
                  << "   B=" << B << " G=" << G << " R=" << R
                  << "   R-G=" << (R - G) << std::endl;
    }
    return 0;
}