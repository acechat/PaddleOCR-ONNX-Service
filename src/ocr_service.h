#ifndef OCR_SERVICE_H
#define OCR_SERVICE_H

#include "ocr_inference.h"
#include <httplib.h>
#include <json.hpp>
#include <string>
#include <mutex>

using json = nlohmann::json;

class OCRService {
public:
    OCRService(const json& service_config);
    void StartServer();
    json Infer(const cv::Mat& img);  // 暴露 for CLI

private:
    std::unique_ptr<OCRInference> inference_;
    json service_config_;
    size_t max_size_;
    size_t request_count_ = 0;
    size_t error_count_ = 0;
    std::mutex metrics_mutex_;

    void ocr_handler(const httplib::Request& req, httplib::Response& res);
    void info_handler(const httplib::Request& req, httplib::Response& res);  // 新增 /info
    json GetInfo();  // 内部：收集版本/模型信息
    std::string base64_decode(const std::string& encoded);
};

#endif // OCR_SERVICE_H