#ifndef OCR_DETECT_H
#define OCR_DETECT_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <json.hpp>
#include <vector>
#include <string>
#include <mutex>

using json = nlohmann::json;

class OCRDetect {
public:
    OCRDetect(const json& det_config);  // 从分层 JSON 初始化
    ~OCRDetect();
    std::vector<std::vector<float>> Detect(const cv::Mat& img);  // 返回 bboxes [x1,y1,x2,y2,score]

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;

    json det_config_;  // 存储子 config
    std::vector<float> mean_, std_;
    bool is_bgr_;
    int min_size_, max_size_;
    float det_threshold_;  // 从 postprocess 层
    float nms_threshold_;  // 从 postprocess 层

    cv::Mat Preprocess(const cv::Mat& img);  // 动态预处理
    std::vector<std::vector<float>> Postprocess(const std::vector<Ort::Value>& outputs, int orig_w, int orig_h, double ratio);
    std::mutex mutex_;  // 线程安全
};

#endif // OCR_DETECT_H