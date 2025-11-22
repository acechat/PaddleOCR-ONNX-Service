#ifndef OCR_INFERENCE_H
#define OCR_INFERENCE_H

#include "ocr_detect.h"
#include "ocr_recognize.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <vector>
#include <memory>
#include <mutex>

using json = nlohmann::json;

struct OCRResult {
    std::vector<float> bbox;
    std::string text;
    float score;
};

class OCRInference {
public:
    OCRInference(const json& service_config);  // 从分层 JSON 初始化
    json Infer(const cv::Mat& img);  // 端到端推理，返回 JSON results array

private:
    std::unique_ptr<OCRDetect> detector_;
    std::unique_ptr<OCRRecognize> recognizer_;
    // std::unique_ptr<OCRCls> cls_;  // 可选方向分类（若启用）

    json service_config_;  // 存储完整 service_config
    std::mutex mutex_;  // 线程安全（全局锁，生产用线程池优化）

    std::vector<OCRResult> RunPipeline(const cv::Mat& img);  // 内部管道
};

#endif // OCR_INFERENCE_H